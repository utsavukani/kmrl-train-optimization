"""
KMRL Database Manager using SQLAlchemy
SIH25081 - Database operations for optimization results and historical data

This module handles:
- SQLite database for development
- Optimization result storage
- Historical data management
- Data retrieval for ML training
"""

import sqlite3
import asyncio
import aiosqlite
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Async database manager for KMRL optimization system
    Handles storage and retrieval of optimization results and train data
    """

    def __init__(self, db_path: str = "kmrl_optimization.db"):
        self.db_path = db_path
        self.connection = None

    async def initialize_database(self):
        """Initialize database tables and schema"""
        try:
            logger.info("Initializing KMRL optimization database...")

            # Create database directory if needed
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(
                self.db_path) else '.', exist_ok=True)

            async with aiosqlite.connect(self.db_path) as db:
                # Create trains table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trains (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        train_id TEXT UNIQUE NOT NULL,
                        last_service_date TEXT,
                        fitness_cert_valid_from TEXT,
                        fitness_cert_valid_to TEXT,
                        jobcard_status TEXT,
                        mileage_since_overhaul INTEGER,
                        iot_sensor_flags TEXT,
                        crew_available BOOLEAN,
                        branding_exposure_hours REAL,
                        stabling_bay TEXT,
                        cleaning_slot TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create optimization_results table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        optimization_id TEXT UNIQUE NOT NULL,
                        request_data TEXT,
                        result_data TEXT,
                        assignments_json TEXT,
                        objectives_achieved TEXT,
                        constraints_log TEXT,
                        ml_predictions TEXT,
                        explanations TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'completed'
                    )
                """)

                # Create historical_performance table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS historical_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        train_id TEXT NOT NULL,
                        date TEXT NOT NULL,
                        assignment TEXT,
                        actual_delay_minutes INTEGER,
                        planned_service_hours REAL,
                        actual_service_hours REAL,
                        maintenance_performed BOOLEAN,
                        issues_reported TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (train_id) REFERENCES trains (train_id)
                    )
                """)

                # Create data_ingestion_log table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS data_ingestion_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_type TEXT,
                        source_file TEXT,
                        records_processed INTEGER,
                        records_validated INTEGER,
                        validation_errors TEXT,
                        ingestion_status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for better performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_trains_id ON trains(train_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_opt_id ON optimization_results(optimization_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_hist_train_date ON historical_performance(train_id, date)")

                await db.commit()
                logger.info("✅ Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    async def store_ingested_data(self, validated_data: List[Dict]) -> Dict[str, Any]:
        """Store ingested and validated train data"""
        try:
            storage_id = f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            async with aiosqlite.connect(self.db_path) as db:
                records_stored = 0

                for record in validated_data:
                    # Insert or update train record
                    await db.execute("""
                        INSERT OR REPLACE INTO trains (
                            train_id, last_service_date, fitness_cert_valid_from,
                            fitness_cert_valid_to, jobcard_status, mileage_since_overhaul,
                            iot_sensor_flags, crew_available, branding_exposure_hours,
                            stabling_bay, cleaning_slot, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.get('train_id'),
                        record.get('last_service_date'),
                        record.get('fitness_cert_valid_from'),
                        record.get('fitness_cert_valid_to'),
                        record.get('jobcard_status'),
                        record.get('mileage_since_overhaul'),
                        record.get('iot_sensor_flags'),
                        record.get('crew_available'),
                        record.get('branding_exposure_hours'),
                        record.get('stabling_bay'),
                        record.get('cleaning_slot'),
                        datetime.now().isoformat()
                    ))
                    records_stored += 1

                # Log ingestion
                await db.execute("""
                    INSERT INTO data_ingestion_log (
                        source_type, records_processed, records_validated, ingestion_status
                    ) VALUES (?, ?, ?, ?)
                """, ('csv_upload', len(validated_data), records_stored, 'success'))

                await db.commit()

                logger.info(f"✅ Stored {records_stored} train records")
                return {
                    'id': storage_id,
                    'records_stored': records_stored,
                    'status': 'success'
                }

        except Exception as e:
            logger.error(f"Data storage failed: {str(e)}")
            return {'id': None, 'error': str(e), 'status': 'failed'}

    async def store_optimization_result(self, optimization_data: Dict) -> Dict[str, Any]:
        """Store optimization result and related data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO optimization_results (
                        optimization_id, request_data, result_data, assignments_json,
                        objectives_achieved, constraints_log, ml_predictions, explanations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    optimization_data.get('optimization_id'),
                    json.dumps(optimization_data.get('input_request', {})),
                    json.dumps(optimization_data.get('result', {})),
                    json.dumps(optimization_data.get(
                        'result', {}).get('assignments', [])),
                    json.dumps(
                        optimization_data.get('result', {}).get('objectives_achieved', {})),
                    json.dumps(optimization_data.get(
                        'result', {}).get('constraints_log', [])),
                    json.dumps(optimization_data.get('predictions', {})),
                    json.dumps(optimization_data.get('explanations', {}))
                ))

                await db.commit()
                logger.info(
                    f"✅ Stored optimization result: {optimization_data.get('optimization_id')}")
                return {'status': 'success', 'optimization_id': optimization_data.get('optimization_id')}

        except Exception as e:
            logger.error(f"Optimization result storage failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

    async def get_optimization_result(self, optimization_id: str) -> Optional[Dict]:
        """Retrieve optimization result by ID"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM optimization_results WHERE optimization_id = ?
                """, (optimization_id,)) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        return {
                            'optimization_id': row[1],
                            'input_request': json.loads(row[2]),
                            'result': json.loads(row[3]),
                            'assignments': json.loads(row[4]),
                            'objectives_achieved': json.loads(row[5]),
                            'constraints_log': json.loads(row[6]),
                            'predictions': json.loads(row[7]),
                            'explanations': json.loads(row[8]),
                            'created_at': row[9]
                        }
                    return None

        except Exception as e:
            logger.error(f"Optimization result retrieval failed: {str(e)}")
            return None

    async def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent optimization history"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT optimization_id, objectives_achieved, created_at, status
                    FROM optimization_results 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,)) as cursor:
                    rows = await cursor.fetchall()

                    history = []
                    for row in rows:
                        history.append({
                            'optimization_id': row[0],
                            'objectives_achieved': json.loads(row[1]) if row[1] else {},
                            'created_at': row[2],
                            'status': row[3]
                        })

                    return history

        except Exception as e:
            logger.error(f"History retrieval failed: {str(e)}")
            return []

    async def get_train_data(self, train_id: Optional[str] = None) -> List[Dict]:
        """Retrieve train data (all trains or specific train)"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if train_id:
                    query = "SELECT * FROM trains WHERE train_id = ?"
                    params = (train_id,)
                else:
                    query = "SELECT * FROM trains ORDER BY train_id"
                    params = ()

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                    trains = []
                    for row in rows:
                        trains.append({
                            'train_id': row[1],
                            'last_service_date': row[2],
                            'fitness_cert_valid_from': row[3],
                            'fitness_cert_valid_to': row[4],
                            'jobcard_status': row[5],
                            'mileage_since_overhaul': row[6],
                            'iot_sensor_flags': row[7],
                            'crew_available': bool(row[8]),
                            'branding_exposure_hours': row[9],
                            'stabling_bay': row[10],
                            'cleaning_slot': row[11],
                            'updated_at': row[13]
                        })

                    return trains

        except Exception as e:
            logger.error(f"Train data retrieval failed: {str(e)}")
            return []

    async def store_historical_performance(self, performance_data: List[Dict]):
        """Store historical performance data for ML training"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for record in performance_data:
                    await db.execute("""
                        INSERT INTO historical_performance (
                            train_id, date, assignment, actual_delay_minutes,
                            planned_service_hours, actual_service_hours, 
                            maintenance_performed, issues_reported
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.get('train_id'),
                        record.get('date'),
                        record.get('assignment'),
                        record.get('actual_delay_minutes'),
                        record.get('planned_service_hours'),
                        record.get('actual_service_hours'),
                        record.get('maintenance_performed'),
                        record.get('issues_reported')
                    ))

                await db.commit()
                logger.info(
                    f"✅ Stored {len(performance_data)} historical performance records")

        except Exception as e:
            logger.error(f"Historical data storage failed: {str(e)}")

    async def get_historical_performance(self, train_id: Optional[str] = None,
                                         days_back: int = 30) -> pd.DataFrame:
        """Retrieve historical performance data for ML training"""
        try:
            start_date = (datetime.now() -
                          timedelta(days=days_back)).strftime('%Y-%m-%d')

            async with aiosqlite.connect(self.db_path) as db:
                if train_id:
                    query = """
                        SELECT * FROM historical_performance 
                        WHERE train_id = ? AND date >= ?
                        ORDER BY date DESC
                    """
                    params = (train_id, start_date)
                else:
                    query = """
                        SELECT * FROM historical_performance 
                        WHERE date >= ?
                        ORDER BY train_id, date DESC
                    """
                    params = (start_date,)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                    # Convert to DataFrame for ML processing
                    columns = ['id', 'train_id', 'date', 'assignment', 'actual_delay_minutes',
                               'planned_service_hours', 'actual_service_hours',
                               'maintenance_performed', 'issues_reported', 'created_at']

                    df = pd.DataFrame(rows, columns=columns)
                    return df

        except Exception as e:
            logger.error(f"Historical performance retrieval failed: {str(e)}")
            return pd.DataFrame()

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}

                # Count records in each table
                for table in ['trains', 'optimization_results', 'historical_performance', 'data_ingestion_log']:
                    async with db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                        count = await cursor.fetchone()
                        stats[f'{table}_count'] = count[0] if count else 0

                # Get latest optimization
                async with db.execute("""
                    SELECT optimization_id, created_at FROM optimization_results 
                    ORDER BY created_at DESC LIMIT 1
                """) as cursor:
                    latest = await cursor.fetchone()
                    stats['latest_optimization'] = {
                        'id': latest[0] if latest else None,
                        'created_at': latest[1] if latest else None
                    }

                # Database file size
                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = round(
                        os.path.getsize(self.db_path) / (1024 * 1024), 2)

                return stats

        except Exception as e:
            logger.error(f"Database stats retrieval failed: {str(e)}")
            return {'error': str(e)}

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Cleanup old data to manage database size"""
        try:
            cutoff_date = (datetime.now() -
                           timedelta(days=days_to_keep)).isoformat()

            async with aiosqlite.connect(self.db_path) as db:
                # Delete old optimization results
                await db.execute("""
                    DELETE FROM optimization_results WHERE created_at < ?
                """, (cutoff_date,))

                # Delete old historical performance data
                await db.execute("""
                    DELETE FROM historical_performance WHERE created_at < ?
                """, (cutoff_date,))

                # Delete old ingestion logs
                await db.execute("""
                    DELETE FROM data_ingestion_log WHERE created_at < ?
                """, (cutoff_date,))

                await db.commit()
                logger.info(
                    f"✅ Cleaned up data older than {days_to_keep} days")

        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")

    async def export_data(self, table_name: str, format: str = 'csv') -> str:
        """Export table data to CSV or JSON format"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(f"SELECT * FROM {table_name}") as cursor:
                    rows = await cursor.fetchall()

                    # Get column names
                    column_names = [description[0]
                                    for description in cursor.description]

                    if format == 'csv':
                        df = pd.DataFrame(rows, columns=column_names)
                        csv_content = df.to_csv(index=False)
                        return csv_content

                    elif format == 'json':
                        records = []
                        for row in rows:
                            record = dict(zip(column_names, row))
                            records.append(record)
                        return json.dumps(records, indent=2)

        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            return f"Export failed: {str(e)}"

    async def close_connections(self):
        """Close database connections on shutdown"""
        logger.info("Closing database connections...")
        # Async sqlite connections are automatically closed
