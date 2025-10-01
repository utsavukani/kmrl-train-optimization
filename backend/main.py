"""
KMRL AI-Driven Train Induction Planning System
SIH25081 - FastAPI Main Application Server

This is the core FastAPI server that orchestrates all optimization components:
- Constraint Programming (CP-SAT)
- Genetic Algorithm (DEAP) 
- Machine Learning Predictions
- LLM Explanations
- What-if Simulations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import our custom modules
from models.cp_sat_solver import ConstraintEngine
from models.genetic_optimizer import MultiObjectiveOptimizer
from models.ml_predictor import DelayPredictor
from models.llm_explainer import ExplanationGenerator
from utils.db_manager import DatabaseManager
from utils.data_validation import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KMRL Train Induction Planning API",
    description="AI-Driven Train Scheduling and Optimization System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
constraint_engine = ConstraintEngine()
genetic_optimizer = MultiObjectiveOptimizer()
ml_predictor = DelayPredictor()
llm_explainer = ExplanationGenerator()
db_manager = DatabaseManager()
data_validator = DataValidator()

# Pydantic models for API requests/responses


class TrainStatus(BaseModel):
    train_id: str
    last_service_date: str
    fitness_cert_valid_from: str
    fitness_cert_valid_to: str
    jobcard_status: str
    mileage_since_overhaul: int
    iot_sensor_flags: str
    crew_available: bool
    branding_exposure_hours: float
    stabling_bay: str
    cleaning_slot: Optional[str] = None


class OptimizationRequest(BaseModel):
    trains: List[TrainStatus]
    target_date: str
    constraints: Dict[str, Any]
    objectives: Dict[str, float]


class SimulationRequest(BaseModel):
    base_optimization_id: str
    modifications: Dict[str, Any]


class OptimizationResult(BaseModel):
    optimization_id: str
    timestamp: str
    assignments: List[Dict[str, Any]]
    objectives_achieved: Dict[str, float]
    conflicts: List[Dict[str, str]]
    explanations: Dict[str, str]

# API Health Check


@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "constraint_engine": "operational",
            "genetic_optimizer": "operational",
            "ml_predictor": "operational",
            "llm_explainer": "operational",
            "database": "operational"
        }
    }


# Backwards-compatible health endpoint used by frontend clients expecting /api/v1/health
@app.get("/api/v1/health")
async def health_check_v1():
    return await health_check()

# Data Ingestion Endpoint


@app.post("/api/v1/ingest")
async def ingest_data(file: UploadFile = File(...), data_type: str = "csv"):
    """
    Ingest train operational data from various sources
    Supports: CSV files, JSON data, Manual entries
    """
    try:
        logger.info(f"Ingesting {data_type} data from file: {file.filename}")

        # Read and validate uploaded file
        content = await file.read()

        if data_type == "csv":
            # Process CSV data (e.g., from Maximo export)
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            validation_result = data_validator.validate_csv_data(df)

        elif data_type == "json":
            # Process JSON data (e.g., from IoT sensors)
            data = json.loads(content.decode('utf-8'))
            validation_result = data_validator.validate_json_data(data)

        else:
            raise HTTPException(
                status_code=400, detail="Unsupported data type")

        # Store validated data in database
        storage_result = await db_manager.store_ingested_data(validation_result['cleaned_data'])

        return {
            "status": "success",
            "message": f"Successfully ingested {validation_result['records_processed']} records",
            "validation_summary": validation_result['summary'],
            "storage_id": storage_result['id']
        }

    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {str(e)}")

# Main Optimization Endpoint


@app.post("/api/v1/optimize", response_model=OptimizationResult)
async def optimize_schedule(request: OptimizationRequest):
    """
    Core optimization endpoint - runs the full pipeline:
    1. CP-SAT constraint filtering
    2. Genetic algorithm optimization
    3. ML predictions
    4. LLM explanations
    """
    try:
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting optimization {optimization_id}")

        # Step 1: Convert request to internal format
        trains_data = [train.dict() for train in request.trains]

        # Step 2: Run constraint programming filter
        logger.info("Running CP-SAT constraint filtering...")
        feasible_solutions = constraint_engine.filter_feasible_assignments(
            trains_data,
            request.constraints
        )

        if not feasible_solutions['feasible']:
            return OptimizationResult(
                optimization_id=optimization_id,
                timestamp=datetime.now().isoformat(),
                assignments=[],
                objectives_achieved={},
                conflicts=feasible_solutions['conflicts'],
                explanations={
                    "error": "No feasible solution found with current constraints"}
            )

        # Step 3: Run genetic algorithm optimization
        logger.info("Running multi-objective genetic algorithm...")
        optimization_result = genetic_optimizer.optimize(
            feasible_solutions['solution_space'],
            request.objectives
        )

        # Defensive: ensure optimization_result is valid
        if not optimization_result or not optimization_result.get('success'):
            logger.error(f"Genetic optimizer failure: {optimization_result}")
            raise HTTPException(
                status_code=500, detail=f"Genetic optimizer failed: {optimization_result.get('error', 'unknown')}")

        # Step 4: Generate ML predictions for selected solution
        logger.info("Generating ML predictions...")
        best_solution = optimization_result.get('best_solution') or []
        # Ensure best_solution is a list of assignments for ML predictor
        if hasattr(best_solution, 'fitness'):
            # If best_solution is an Individual object, convert to list
            try:
                best_solution_list = list(best_solution)
            except Exception:
                best_solution_list = []
        else:
            best_solution_list = best_solution

        predictions = ml_predictor.predict_performance(
            best_solution_list
        )

        # Step 5: Generate LLM explanations
        logger.info("Generating explanations...")
        explanations = llm_explainer.generate_explanations(
            best_solution_list,
            {
                'constraints_applied': feasible_solutions.get('constraints_log', []),
                'optimization_metrics': optimization_result.get('metrics', {}),
                'ml_predictions': predictions
            }
        )

        # Step 6: Store optimization result
        storage_result = await db_manager.store_optimization_result({
            'optimization_id': optimization_id,
            'input_request': request.dict(),
            'result': optimization_result,
            'predictions': predictions,
            'explanations': explanations
        })

        return OptimizationResult(
            optimization_id=optimization_id,
            timestamp=datetime.now().isoformat(),
            assignments=optimization_result['assignments'],
            objectives_achieved=optimization_result['objectives_achieved'],
            conflicts=[],
            explanations=explanations
        )

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Optimization failed: {str(e)}")

# What-if Simulation Endpoint


@app.post("/api/v1/simulate")
async def whatif_simulation(request: SimulationRequest):
    """
    Run what-if simulation by modifying constraints and re-optimizing
    """
    try:
        logger.info(
            f"Running what-if simulation based on {request.base_optimization_id}")

        # Retrieve base optimization
        base_optimization = await db_manager.get_optimization_result(
            request.base_optimization_id
        )

        if not base_optimization:
            raise HTTPException(
                status_code=404, detail="Base optimization not found")

        # Apply modifications to create new scenario
        modified_request = base_optimization['input_request'].copy()

        # Apply user modifications (e.g., force train to maintenance, change crew)
        for modification_key, modification_value in request.modifications.items():
            if modification_key.startswith('train_'):
                train_id = modification_key.split('_')[1]
                # Find and modify specific train
                for train in modified_request['trains']:
                    if train['train_id'] == train_id:
                        train.update(modification_value)
            else:
                # Global constraint modification
                modified_request['constraints'][modification_key] = modification_value

        # Re-run optimization with modifications
        simulation_request = OptimizationRequest(**modified_request)
        simulation_result = await optimize_schedule(simulation_request)

        # Calculate deltas compared to base optimization
        base_result = base_optimization['result']
        delta_analysis = {
            'objective_changes': {},
            'assignment_changes': [],
            'impact_summary': {}
        }

        # Calculate objective deltas
        for objective, base_value in base_result['objectives_achieved'].items():
            new_value = simulation_result.objectives_achieved.get(objective, 0)
            delta_analysis['objective_changes'][objective] = new_value - base_value

        # Add delta analysis to result
        simulation_result.explanations['delta_analysis'] = delta_analysis

        return simulation_result

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Simulation failed: {str(e)}")

# ML Prediction Endpoint


@app.post("/api/v1/predict")
async def predict_performance(trains: List[TrainStatus]):
    """
    Generate ML predictions for delay risk and maintenance urgency
    """
    try:
        logger.info(f"Generating predictions for {len(trains)} trains")

        predictions = []
        for train in trains:
            train_prediction = ml_predictor.predict_individual_train(
                train.dict())
            predictions.append({
                'train_id': train.train_id,
                'delay_risk_probability': train_prediction['delay_risk'],
                'maintenance_urgency_score': train_prediction['maintenance_urgency'],
                'confidence_level': train_prediction['confidence'],
                'key_risk_factors': train_prediction['risk_factors']
            })

        return {
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'model_version': ml_predictor.get_model_version()
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")

# Get Optimization History


@app.get("/api/v1/optimizations")
async def get_optimization_history(limit: int = 10):
    """
    Retrieve historical optimization results
    """
    try:
        history = await db_manager.get_optimization_history(limit=limit)
        return {
            'optimizations': history,
            'count': len(history)
        }
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"History retrieval failed: {str(e)}")

# Export Results Endpoint


@app.get("/api/v1/export/{optimization_id}")
async def export_optimization_result(optimization_id: str, format: str = "json"):
    """
    Export optimization results in various formats (JSON, CSV, PDF)
    """
    try:
        result = await db_manager.get_optimization_result(optimization_id)

        if not result:
            raise HTTPException(
                status_code=404, detail="Optimization result not found")

        if format == "csv":
            # Convert to CSV format
            df = pd.DataFrame(result['result']['assignments'])
            csv_content = df.to_csv(index=False)
            return {"format": "csv", "content": csv_content}

        elif format == "json":
            return {"format": "json", "content": result}

        else:
            raise HTTPException(
                status_code=400, detail="Unsupported export format")

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Initialize database on startup


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    logger.info("🚄 Starting KMRL Optimization System...")

    # Initialize database
    await db_manager.initialize_database()

    # Load ML models
    await ml_predictor.load_models()

    # Initialize LLM
    await llm_explainer.initialize()

    logger.info("✅ All components initialized successfully!")

# Shutdown handler


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down KMRL Optimization System...")
    await db_manager.close_connections()

# Run the server
if __name__ == "__main__":
    import io
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
