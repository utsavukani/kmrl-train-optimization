"""
KMRL Synthetic Dataset Generator
SIH25081 - Generate realistic train operational data for development and demo

This module generates:
- 25 KMRL trains with realistic operational patterns
- 180 days of historical data
- Maintenance cycles, certificate renewals, job cards
- Edge cases for comprehensive testing
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate realistic synthetic data for KMRL train optimization system
    Based on real operational patterns and industry standards
    """

    def __init__(self):
        self.num_trains = 25
        self.historical_days = 180
        self.train_ids = [
            f"KMRL-{str(i).zfill(3)}" for i in range(1, self.num_trains + 1)]

        # Realistic operational parameters based on metro systems
        self.maintenance_cycle_days = 30
        self.certificate_validity_days = 90
        self.overhaul_mileage = 100000
        self.daily_avg_km = 250
        self.service_hours_per_day = 16

        # Depot configuration
        self.total_bays = 12
        self.maintenance_bays = 4
        self.cleaning_bays = 3

        logging.basicConfig(level=logging.INFO)
        logger.info("Synthetic data generator initialized for KMRL system")

    def generate_complete_dataset(self) -> Dict[str, Any]:
        """
        Generate complete dataset including current status and historical data

        Returns:
            Complete dataset with trains, historical performance, and metadata
        """
        try:
            logger.info(
                f"Generating synthetic dataset for {self.num_trains} trains over {self.historical_days} days")

            # Generate current train status
            current_trains = self.generate_current_train_status()

            # Generate historical performance data
            historical_data = self.generate_historical_performance()

            # Generate job card data
            job_cards = self.generate_job_card_data()

            # Generate IoT sensor data
            iot_data = self.generate_iot_sensor_data()

            # Generate branding contracts
            branding_data = self.generate_branding_contracts()

            complete_dataset = {
                'trains': current_trains,
                'historical_performance': historical_data,
                'job_cards': job_cards,
                'iot_sensors': iot_data,
                'branding_contracts': branding_data,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'num_trains': self.num_trains,
                    'historical_days': self.historical_days,
                    'data_version': '1.0.0',
                    'generator': 'KMRL Synthetic Data Generator'
                }
            }

            logger.info("✅ Complete synthetic dataset generated successfully")
            return complete_dataset

        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}")
            raise

    def generate_current_train_status(self) -> List[Dict[str, Any]]:
        """
        Generate current operational status for all trains
        """
        trains = []
        today = datetime.now()

        for i, train_id in enumerate(self.train_ids):
            # Vary service patterns to create realistic scenarios
            days_since_service = random.randint(1, 45)
            last_service_date = today - timedelta(days=days_since_service)

            # Certificate validity (some expiring soon, some expired)
            cert_issued_days_ago = random.randint(1, 120)
            cert_valid_from = today - timedelta(days=cert_issued_days_ago)
            cert_valid_to = cert_valid_from + \
                timedelta(days=self.certificate_validity_days)

            # Mileage based on age and usage pattern
            base_mileage = random.randint(20000, 95000)
            additional_mileage = days_since_service * random.randint(200, 300)
            total_mileage = base_mileage + additional_mileage

            # Job card status (weighted for realism)
            jobcard_statuses = ['NONE', 'OPEN',
                                'CLOSED', 'CRITICAL_OPEN', 'PENDING']
            jobcard_weights = [0.4, 0.3, 0.2, 0.05, 0.05]
            jobcard_status = np.random.choice(
                jobcard_statuses, p=jobcard_weights)

            # IoT sensor flags
            sensor_flags = self._generate_sensor_flags(
                total_mileage, days_since_service)

            # Crew availability (mostly available, some exceptions)
            crew_available = random.random() > 0.1  # 90% availability

            # Branding exposure requirements
            branding_hours = round(random.uniform(8, 18), 1)

            # Stabling bay assignment
            stabling_bay = f"BAY-{random.randint(1, self.total_bays):02d}"

            # Cleaning slot requirements
            cleaning_slots = ['NONE', 'SCHEDULED', 'IN_PROGRESS', 'REQUIRED']
            cleaning_weights = [0.5, 0.2, 0.1, 0.2]
            cleaning_slot = np.random.choice(
                cleaning_slots, p=cleaning_weights)

            train_data = {
                'train_id': train_id,
                'last_service_date': last_service_date.strftime('%Y-%m-%d'),
                'fitness_cert_valid_from': cert_valid_from.strftime('%Y-%m-%d'),
                'fitness_cert_valid_to': cert_valid_to.strftime('%Y-%m-%d'),
                'jobcard_status': jobcard_status,
                'mileage_since_overhaul': total_mileage,
                'iot_sensor_flags': sensor_flags,
                'crew_available': crew_available,
                'branding_exposure_hours': branding_hours,
                'stabling_bay': stabling_bay,
                'cleaning_slot': cleaning_slot,
                'created_at': today.isoformat(),
                'updated_at': today.isoformat()
            }

            trains.append(train_data)

        # Ensure we have realistic constraints
        trains = self._apply_realistic_constraints(trains)

        logger.info(f"Generated current status for {len(trains)} trains")
        return trains

    def generate_historical_performance(self) -> List[Dict[str, Any]]:
        """Generate historical performance data for ML training"""

        historical_data = []

        for train_id in self.train_ids:
            for days_back in range(1, self.historical_days + 1):
                date = (datetime.now() - timedelta(days=days_back)
                        ).strftime('%Y-%m-%d')

                # Assignment pattern (more service during weekdays)
                is_weekend = (datetime.now() -
                              timedelta(days=days_back)).weekday() >= 5
                assignment_probs = [
                    0.7, 0.2, 0.1] if not is_weekend else [0.5, 0.3, 0.2]
                assignment = np.random.choice(
                    ['SERVICE', 'STANDBY', 'MAINTENANCE'], p=assignment_probs)

                # Delay patterns (influenced by various factors)
                if assignment == 'SERVICE':
                    delay_factors = {
                        'weather': random.uniform(0, 3),
                        'technical': random.uniform(0, 4),
                        'operational': random.uniform(0, 2)
                    }

                    total_delay = sum(delay_factors.values())
                    actual_delay_minutes = int(
                        max(0, random.uniform(0, 5) + total_delay))

                    planned_hours = random.uniform(14, 18)
                    actual_hours = planned_hours - (actual_delay_minutes / 60)
                    actual_hours = max(8, actual_hours)
                else:
                    actual_delay_minutes = 0
                    planned_hours = 0
                    actual_hours = 0

                maintenance_performed = (assignment == 'MAINTENANCE')

                issues_list = []
                if assignment == 'SERVICE' and random.random() < 0.1:
                    possible_issues = ['Door malfunction', 'HVAC issue', 'Brake squeal',
                                       'Minor electrical issue', 'Communication problem']
                    issues_list.append(random.choice(possible_issues))
                if assignment == 'MAINTENANCE' and random.random() < 0.3:
                    maintenance_issues = ['Scheduled brake pad replacement', 'Motor inspection',
                                          'Software update', 'Safety system check', 'Cleaning and lubrication']
                    issues_list.append(random.choice(maintenance_issues))

                issues_reported = ', '.join(
                    issues_list) if issues_list else 'None'

                historical_record = {
                    'train_id': train_id,
                    'date': date,
                    'assignment': assignment,
                    'actual_delay_minutes': actual_delay_minutes,
                    'planned_service_hours': round(planned_hours, 2),
                    'actual_service_hours': round(actual_hours, 2),
                    'maintenance_performed': maintenance_performed,
                    'issues_reported': issues_reported,
                    'created_at': datetime.now().isoformat()
                }

                historical_data.append(historical_record)

        logger.info(
            f"Generated {len(historical_data)} historical performance records")
        return historical_data

    def generate_job_card_data(self) -> List[Dict[str, Any]]:
        """Generate realistic job card data"""

        job_cards = []
        job_card_id = 1000

        for train_id in self.train_ids:
            num_cards = random.choices([0, 1, 2, 3], weights=[
                                       0.3, 0.4, 0.2, 0.1])[0]

            for _ in range(num_cards):
                job_types = ['Brake System Inspection', 'Door Mechanism Service', 'HVAC Maintenance',
                             'Electrical System Check', 'Software Update', 'Safety System Verification',
                             'Cleaning and Sanitization', 'Tire Inspection', 'Communication System Test']

                job_type = random.choice(job_types)
                priorities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                priority_weights = [0.4, 0.3, 0.2, 0.1]
                priority = np.random.choice(priorities, p=priority_weights)
                statuses = ['OPEN', 'IN_PROGRESS', 'CLOSED', 'SCHEDULED']
                status_weights = [0.3, 0.2, 0.4, 0.1]
                status = np.random.choice(statuses, p=status_weights)
                estimated_hours = random.uniform(0.5, 8)
                created_days_ago = random.randint(1, 30)
                created_date = datetime.now() - timedelta(days=created_days_ago)

                job_card = {
                    'job_card_id': f"JC{job_card_id:06d}",
                    'train_id': train_id,
                    'job_type': job_type,
                    'priority': priority,
                    'status': status,
                    'estimated_hours': round(estimated_hours, 2),
                    'created_date': created_date.strftime('%Y-%m-%d'),
                    'due_date': (created_date + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'assigned_technician': f"TECH-{random.randint(1, 20):03d}",
                    'description': f"{job_type} for {train_id} - Priority: {priority}"
                }

                job_cards.append(job_card)
                job_card_id += 1

        logger.info(f"Generated {len(job_cards)} job card records")
        return job_cards

    def generate_iot_sensor_data(self) -> List[Dict[str, Any]]:
        """Generate IoT sensor readings"""

        sensor_data = []

        for train_id in self.train_ids:
            sensor_types = ['brake_temperature', 'motor_temperature', 'hvac_efficiency',
                            'door_cycle_count', 'vibration_level', 'battery_voltage',
                            'air_pressure', 'wheel_wear_indicator']

            for sensor_type in sensor_types:
                if sensor_type == 'brake_temperature':
                    value = random.uniform(45, 85)
                    threshold = 80
                elif sensor_type == 'motor_temperature':
                    value = random.uniform(60, 90)
                    threshold = 85
                elif sensor_type == 'hvac_efficiency':
                    value = random.uniform(75, 95)
                    threshold = 80
                elif sensor_type == 'door_cycle_count':
                    value = random.randint(50000, 200000)
                    threshold = 180000
                elif sensor_type == 'vibration_level':
                    value = random.uniform(0.5, 3.0)
                    threshold = 2.5
                elif sensor_type == 'battery_voltage':
                    value = random.uniform(110, 125)
                    threshold = 115
                elif sensor_type == 'air_pressure':
                    value = random.uniform(8.0, 9.5)
                    threshold = 8.2
                elif sensor_type == 'wheel_wear_indicator':
                    value = random.uniform(0, 100)
                    threshold = 80
                else:
                    value = random.uniform(0, 100)
                    threshold = 80

                if sensor_type in ['hvac_efficiency', 'battery_voltage', 'air_pressure']:
                    status = 'CRITICAL' if value < threshold else 'NORMAL'
                else:
                    status = 'CRITICAL' if value > threshold else 'NORMAL'

                if status == 'NORMAL' and random.random() < 0.1:
                    status = 'WARNING'

                sensor_record = {
                    'train_id': train_id,
                    'sensor_type': sensor_type,
                    'sensor_value': round(value, 2),
                    'threshold_value': threshold,
                    'status': status,
                    'unit': self._get_sensor_unit(sensor_type),
                    'last_reading': datetime.now().isoformat(),
                    'location': random.choice(['CAR_1', 'CAR_2', 'CAR_3', 'CAR_4'])
                }

                sensor_data.append(sensor_record)

        logger.info(f"Generated {len(sensor_data)} IoT sensor readings")
        return sensor_data

    def generate_branding_contracts(self) -> List[Dict[str, Any]]:
        """Generate branding/advertising contract data"""

        contracts = []
        contract_id = 2000

        advertisers = ['Kerala Tourism', 'Federal Bank', 'Malabar Gold', 'Kalyan Jewellers',
                       'Lulu Mall', 'Reliance Digital', 'Amazon', 'Flipkart', 'BSNL', 'Airtel']

        for train_id in self.train_ids:
            num_contracts = random.choices(
                [0, 1, 2], weights=[0.2, 0.6, 0.2])[0]

            for _ in range(num_contracts):
                advertiser = random.choice(advertisers)
                start_date = datetime.now() - timedelta(days=random.randint(10, 90))
                contract_duration = random.choice([30, 60, 90, 180])
                end_date = start_date + timedelta(days=contract_duration)
                min_daily_hours = random.uniform(8, 16)
                min_weekly_exposure = min_daily_hours * 7
                daily_rate = random.uniform(5000, 15000)
                total_value = daily_rate * contract_duration

                contract = {
                    'contract_id': f"BC{contract_id:06d}",
                    'train_id': train_id,
                    'advertiser': advertiser,
                    'campaign_name': f"{advertiser} Campaign {random.randint(1, 20)}",
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'min_daily_exposure_hours': round(min_daily_hours, 1),
                    'min_weekly_exposure_hours': round(min_weekly_exposure, 1),
                    'daily_rate_inr': round(daily_rate, 2),
                    'total_contract_value_inr': round(total_value, 2),
                    'penalty_rate_per_hour': round(daily_rate * 0.1, 2),
                    'status': 'ACTIVE' if end_date > datetime.now() else 'EXPIRED',
                    'branding_location': random.choice(['EXTERIOR_FULL', 'EXTERIOR_PARTIAL', 'INTERIOR'])
                }

                contracts.append(contract)
                contract_id += 1

        logger.info(f"Generated {len(contracts)} branding contracts")
        return contracts

    def save_dataset_to_files(self, dataset: Dict[str, Any], output_dir: str = "."):
        """Save dataset to separate CSV files"""

        import os
        os.makedirs(output_dir, exist_ok=True)

        trains_df = pd.DataFrame(dataset['trains'])
        trains_df.to_csv(f"{output_dir}/synthetic_trains.csv", index=False)

        historical_df = pd.DataFrame(dataset['historical_performance'])
        historical_df.to_csv(
            f"{output_dir}/synthetic_historical_performance.csv", index=False)

        if dataset['job_cards']:
            jobcards_df = pd.DataFrame(dataset['job_cards'])
            jobcards_df.to_csv(
                f"{output_dir}/synthetic_job_cards.csv", index=False)

        iot_df = pd.DataFrame(dataset['iot_sensors'])
        iot_df.to_csv(f"{output_dir}/synthetic_iot_sensors.csv", index=False)

        if dataset['branding_contracts']:
            branding_df = pd.DataFrame(dataset['branding_contracts'])
            branding_df.to_csv(
                f"{output_dir}/synthetic_branding_contracts.csv", index=False)

        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)

        logger.info(f"✅ Dataset saved to {output_dir}/")

    def _generate_sensor_flags(self, mileage: int, days_since_service: int) -> str:
        """Generate realistic IoT sensor flags based on train condition"""

        flags = []

        if mileage > 90000 and random.random() < 0.3:
            flags.append('HIGH_MILEAGE_ALERT')

        if days_since_service > 35 and random.random() < 0.2:
            flags.append('SERVICE_DUE')

        sensor_alerts = [
            'BRAKE_TEMP_HIGH', 'MOTOR_VIBRATION', 'HVAC_EFFICIENCY_LOW',
            'DOOR_MALFUNCTION', 'BATTERY_LOW', 'AIR_PRESSURE_LOW'
        ]

        for alert in sensor_alerts:
            if random.random() < 0.05:
                flags.append(alert)

        if random.random() < 0.02:
            critical_alerts = ['CRITICAL_BRAKE_FAILURE',
                               'CRITICAL_MOTOR_FAULT', 'CRITICAL_SAFETY_SYSTEM']
            flags.append(random.choice(critical_alerts))

        return ', '.join(flags) if flags else 'NORMAL'

    def _apply_realistic_constraints(self, trains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply realistic constraints to ensure valid scenarios"""

        service_capable_count = sum(
            self._is_service_capable(t) for t in trains)

        if service_capable_count < 18:
            trains_to_fix = 18 - service_capable_count
            non_capable_trains = [
                t for t in trains if not self._is_service_capable(t)]

            for i in range(min(trains_to_fix, len(non_capable_trains))):
                train = non_capable_trains[i]
                train['fitness_cert_valid_to'] = (
                    datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                if 'CRITICAL' in train['jobcard_status']:
                    train['jobcard_status'] = 'OPEN'
                train['iot_sensor_flags'] = train['iot_sensor_flags'].replace(
                    'CRITICAL', 'WARNING')

        return trains

    def _is_service_capable(self, train: Dict[str, Any]) -> bool:
        """Check if train is capable of service assignment"""

        try:
            cert_valid_to = datetime.strptime(
                train['fitness_cert_valid_to'], '%Y-%m-%d')
            if cert_valid_to < datetime.now():
                return False
        except:
            return False

        if 'CRITICAL' in train['jobcard_status']:
            return False

        if 'CRITICAL' in train['iot_sensor_flags']:
            return False

        return True

    def _get_sensor_unit(self, sensor_type: str) -> str:
        """Get appropriate unit for sensor type"""

        units = {
            'brake_temperature': '°C',
            'motor_temperature': '°C',
            'hvac_efficiency': '%',
            'door_cycle_count': 'cycles',
            'vibration_level': 'mm/s',
            'battery_voltage': 'V',
            'air_pressure': 'bar',
            'wheel_wear_indicator': '%'
        }

        return units.get(sensor_type, 'units')


def main():
    """Generate synthetic dataset when run directly"""

    logging.basicConfig(level=logging.INFO)

    print("🚄 KMRL Synthetic Data Generator")
    print("=" * 50)

    generator = SyntheticDataGenerator()

    print("Generating complete synthetic dataset...")
    dataset = generator.generate_complete_dataset()

    print("Saving dataset to CSV files...")
    generator.save_dataset_to_files(dataset, output_dir=".")

    print("\n✅ Synthetic dataset generation completed!")
    print(f"Generated data for {len(dataset['trains'])} trains")
    print(f"Historical data: {len(dataset['historical_performance'])} records")
    print(f"Job cards: {len(dataset['job_cards'])} records")
    print(f"IoT sensors: {len(dataset['iot_sensors'])} readings")
    print(
        f"Branding contracts: {len(dataset['branding_contracts'])} contracts")

    print("\n📊 Sample Train Data:")
    for i, train in enumerate(dataset['trains'][:3]):
        print(
            f"  {i+1}. {train['train_id']}: {train['jobcard_status']} - Mileage: {train['mileage_since_overhaul']} km")

    print(f"\nFiles saved in './' directory")


if __name__ == "__main__":
    main()
