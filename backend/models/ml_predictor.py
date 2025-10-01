"""
KMRL Machine Learning Predictor for Train Performance
SIH25081 - XGBoost-based delay and maintenance prediction

This module provides ML-powered predictions for:
- Delay risk probability
- Maintenance urgency scoring
- Performance degradation forecasting
- Component failure prediction
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import xgboost as xgb

logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    ML-based predictor for train delays and maintenance requirements
    Uses XGBoost and Random Forest for robust predictions
    """

    def __init__(self):
        self.delay_model = None
        self.maintenance_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_version = "1.0.0"
        self.is_trained = False

    async def load_models(self):
        """Load pre-trained ML models or train new ones"""
        try:
            logger.info(
                "Loading ML models for delay and maintenance prediction...")

            # Try to load existing models
            try:
                self.delay_model = joblib.load('models/delay_predictor.pkl')
                self.maintenance_model = joblib.load(
                    'models/maintenance_predictor.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                self.feature_columns = joblib.load(
                    'models/feature_columns.pkl')
                self.is_trained = True
                logger.info("✅ Pre-trained models loaded successfully")
            except FileNotFoundError:
                logger.info(
                    "🔄 No pre-trained models found. Training new models...")
                await self.train_models()

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Create dummy models for demonstration
            await self.create_dummy_models()

    async def train_models(self):
        """Train ML models using synthetic historical data"""
        try:
            logger.info("Training ML models on synthetic data...")

            # Generate synthetic training data
            training_data = self.generate_synthetic_training_data()

            # Prepare features and targets
            X, y_delay, y_maintenance = self.prepare_training_data(
                training_data)

            # Split data
            X_train, X_test, y_delay_train, y_delay_test = train_test_split(
                X, y_delay, test_size=0.2, random_state=42
            )
            _, _, y_maint_train, y_maint_test = train_test_split(
                X, y_maintenance, test_size=0.2, random_state=42
            )

            # Train delay prediction model (classification)
            self.delay_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.delay_model.fit(X_train, y_delay_train)

            # Train maintenance urgency model (regression)
            self.maintenance_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.maintenance_model.fit(X_train, y_maint_train)

            # Evaluate models
            delay_accuracy = accuracy_score(
                y_delay_test, self.delay_model.predict(X_test))
            maintenance_mse = mean_squared_error(
                y_maint_test, self.maintenance_model.predict(X_test))

            logger.info(
                f"✅ Models trained - Delay Accuracy: {delay_accuracy:.3f}, Maintenance MSE: {maintenance_mse:.3f}")

            # Save models
            await self.save_models()
            self.is_trained = True

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            await self.create_dummy_models()

    def generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate realistic synthetic training data"""

        np.random.seed(42)
        n_samples = 5000

        # Generate base features
        data = {
            'mileage_since_overhaul': np.random.normal(50000, 20000, n_samples).clip(0, 120000),
            'days_since_last_service': np.random.exponential(15, n_samples).clip(1, 90),
            'fitness_cert_days_remaining': np.random.normal(30, 15, n_samples).clip(0, 90),
            'open_jobcards_count': np.random.poisson(2, n_samples),
            'critical_jobcards_count': np.random.poisson(0.3, n_samples),
            'service_hours_last_week': np.random.normal(80, 25, n_samples).clip(0, 168),
            'average_daily_km': np.random.normal(250, 50, n_samples).clip(100, 400),
            'brake_pad_wear_percent': np.random.normal(60, 25, n_samples).clip(0, 100),
            'hvac_efficiency_percent': np.random.normal(85, 15, n_samples).clip(30, 100),
            'motor_temperature_avg': np.random.normal(75, 10, n_samples).clip(50, 95),
            'weather_risk_score': np.random.uniform(0, 1, n_samples),
            'rush_hour_usage_percent': np.random.normal(40, 15, n_samples).clip(0, 100),
            'weekend_usage': np.random.binomial(1, 0.3, n_samples),
            'seasonal_factor': np.random.uniform(0.8, 1.2, n_samples),
        }

        df = pd.DataFrame(data)

        # Generate target variables based on logical relationships

        # Delay probability (higher with more mileage, older service, critical issues)
        delay_risk = (
            0.3 * (df['mileage_since_overhaul'] / 120000) +
            0.2 * (df['days_since_last_service'] / 90) +
            0.25 * (df['critical_jobcards_count'] / 5) +
            0.15 * (1 - df['hvac_efficiency_percent'] / 100) +
            0.1 * df['weather_risk_score'] +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0, 1)

        df['delay_risk_binary'] = (delay_risk > 0.4).astype(int)

        # Maintenance urgency score (0-100)
        maintenance_urgency = (
            40 * (df['mileage_since_overhaul'] / 120000) +
            20 * (df['brake_pad_wear_percent'] / 100) +
            15 * (df['days_since_last_service'] / 90) +
            10 * (df['critical_jobcards_count'] / 5) +
            10 * (1 - df['hvac_efficiency_percent'] / 100) +
            5 * (df['motor_temperature_avg'] / 100) +
            np.random.normal(0, 5, n_samples)
        ).clip(0, 100)

        df['maintenance_urgency_score'] = maintenance_urgency

        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""

        # Feature columns (excluding targets)
        self.feature_columns = [col for col in df.columns
                                if col not in ['delay_risk_binary', 'maintenance_urgency_score']]

        X = df[self.feature_columns].values
        y_delay = df['delay_risk_binary'].values
        y_maintenance = df['maintenance_urgency_score'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y_delay, y_maintenance

    async def create_dummy_models(self):
        """Create simple dummy models for demonstration"""
        logger.info("Creating dummy models for demonstration...")

        # Simple rule-based models
        self.delay_model = DummyDelayModel()
        self.maintenance_model = DummyMaintenanceModel()
        self.is_trained = True

        logger.info("✅ Dummy models created successfully")

    async def save_models(self):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs('models', exist_ok=True)

            joblib.dump(self.delay_model, 'models/delay_predictor.pkl')
            joblib.dump(self.maintenance_model,
                        'models/maintenance_predictor.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            joblib.dump(self.feature_columns, 'models/feature_columns.pkl')

            logger.info("✅ Models saved successfully")
        except Exception as e:
            logger.warning(f"Model saving failed: {str(e)}")

    def predict_performance(self, assignments: List[Dict]) -> Dict[str, Any]:
        """Generate predictions for a list of train assignments"""

        if not self.is_trained:
            return {'error': 'Models not trained yet', 'predictions': []}

        try:
            predictions = []

            for assignment in assignments:
                train_prediction = self.predict_individual_train(assignment)
                predictions.append(train_prediction)
            # Aggregate predictions (guard against empty predictions)
            if len(predictions) == 0:
                total_delay_risk = 0.0
            else:
                total_delay_risk = sum(p.get('delay_risk', 0.0)
                                       for p in predictions) / max(1, len(predictions))
            high_maintenance_count = sum(
                1 for p in predictions if p['maintenance_urgency'] > 70)

            return {
                'individual_predictions': predictions,
                'aggregate_metrics': {
                    'average_delay_risk': total_delay_risk,
                    'high_maintenance_trains': high_maintenance_count,
                    'fleet_reliability_score': max(0, 100 - total_delay_risk * 100 - high_maintenance_count * 5)
                },
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e), 'predictions': []}

    def predict_individual_train(self, train_data: Dict) -> Dict[str, Any]:
        """Generate predictions for individual train"""

        try:
            # Extract features from train data
            features = self.extract_features(train_data)

            if isinstance(self.delay_model, DummyDelayModel):
                # Use dummy models
                delay_risk = self.delay_model.predict(features)
                maintenance_urgency = self.maintenance_model.predict(features)
            else:
                # Use trained ML models
                features_scaled = self.scaler.transform([features])
                delay_risk = self.delay_model.predict_proba(
                    features_scaled)[0][1]  # Probability of delay
                maintenance_urgency = self.maintenance_model.predict(features_scaled)[
                    0]

            # Generate risk factors explanation
            risk_factors = self.identify_risk_factors(train_data, features)

            # Calculate confidence based on feature quality
            confidence = self.calculate_prediction_confidence(train_data)

            return {
                'train_id': train_data.get('train_id', 'unknown'),
                'delay_risk': min(1.0, max(0.0, float(delay_risk))),
                'maintenance_urgency': min(100.0, max(0.0, float(maintenance_urgency))),
                'confidence': confidence,
                'risk_factors': risk_factors,
                'prediction_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Individual prediction failed: {str(e)}")
            return {
                'train_id': train_data.get('train_id', 'unknown'),
                'delay_risk': 0.1,  # Conservative default
                'maintenance_urgency': 20.0,
                'confidence': 0.5,
                'risk_factors': ['prediction_error'],
                'error': str(e)
            }

    def extract_features(self, train_data: Dict) -> List[float]:
        """Extract numerical features from train data"""

        # Calculate derived features
        last_service_str = train_data.get('last_service_date') or '2024-01-01'
        try:
            last_service = datetime.strptime(last_service_str, '%Y-%m-%d')
        except Exception:
            last_service = datetime.now() - timedelta(days=30)
        days_since_service = max(1, (datetime.now() - last_service).days)

        cert_valid_to_str = train_data.get(
            'fitness_cert_valid_to') or '2024-12-31'
        try:
            cert_valid_to = datetime.strptime(cert_valid_to_str, '%Y-%m-%d')
        except Exception:
            cert_valid_to = datetime.now() + timedelta(days=90)
        cert_days_remaining = max(0, (cert_valid_to - datetime.now()).days)

        # Extract features matching training data
        features = [
            train_data.get('mileage_since_overhaul', 50000),
            days_since_service,
            cert_days_remaining,
            len(train_data.get('jobcard_status', '').split(',')
                ) if train_data.get('jobcard_status') else 0,
            1 if 'CRITICAL' in train_data.get('jobcard_status', '') else 0,
            train_data.get('service_hours', 80),
            (train_data.get('mileage_since_overhaul', 50000) /
             days_since_service) if days_since_service > 0 else 250,
            70.0,  # brake_pad_wear_percent (mock)
            85.0,  # hvac_efficiency_percent (mock)
            75.0,  # motor_temperature_avg (mock)
            0.3,   # weather_risk_score (mock)
            40.0,  # rush_hour_usage_percent (mock)
            0,     # weekend_usage (mock)
            1.0,   # seasonal_factor (mock)
        ]

        return features

    def identify_risk_factors(self, train_data: Dict, features: List[float]) -> List[str]:
        """Identify key risk factors for this train"""

        risk_factors = []

        # High mileage
        if features[0] > 80000:
            risk_factors.append('high_mileage')

        # Long time since service
        if features[1] > 30:
            risk_factors.append('overdue_service')

        # Certificate expiring soon
        if features[2] < 7:
            risk_factors.append('certificate_expiring')

        # Critical job cards
        if features[4] > 0:
            risk_factors.append('critical_maintenance_issues')

        # Heavy usage
        if features[5] > 100:
            risk_factors.append('heavy_usage_pattern')

        if not risk_factors:
            risk_factors.append('normal_operation')

        return risk_factors

    def calculate_prediction_confidence(self, train_data: Dict) -> float:
        """Calculate confidence level for predictions"""

        confidence = 0.8  # Base confidence

        # Reduce confidence for missing data
        if not train_data.get('last_service_date'):
            confidence -= 0.2
        if not train_data.get('mileage_since_overhaul'):
            confidence -= 0.1
        if not train_data.get('jobcard_status'):
            confidence -= 0.1

        # Increase confidence for recent data
        try:
            last_service = datetime.strptime(train_data.get(
                'last_service_date', '2024-01-01'), '%Y-%m-%d')
            days_old = (datetime.now() - last_service).days
            if days_old < 7:
                confidence += 0.1
        except:
            pass

        return min(1.0, max(0.1, confidence))

    def get_model_version(self) -> str:
        """Get current model version"""
        return self.model_version

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""

        if not self.is_trained or isinstance(self.delay_model, DummyDelayModel):
            return {'error': 'Model not trained or using dummy model'}

        try:
            # Get feature importance from delay model
            importance = self.delay_model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))

            return feature_importance
        except Exception as e:
            return {'error': str(e)}


class DummyDelayModel:
    """Simple rule-based delay prediction for demonstration"""

    def predict(self, features: List[float]) -> float:
        """Predict delay risk based on simple rules"""
        mileage = features[0]
        days_since_service = features[1]
        cert_days_remaining = features[2]
        critical_issues = features[4]

        risk = 0.1  # Base risk

        # Increase risk based on factors
        if mileage > 80000:
            risk += 0.3
        elif mileage > 60000:
            risk += 0.2

        if days_since_service > 45:
            risk += 0.25
        elif days_since_service > 30:
            risk += 0.15

        if cert_days_remaining < 7:
            risk += 0.2

        if critical_issues > 0:
            risk += 0.3

        return min(1.0, risk)


class DummyMaintenanceModel:
    """Simple rule-based maintenance urgency for demonstration"""

    def predict(self, features: List[float]) -> float:
        """Predict maintenance urgency based on simple rules"""
        mileage = features[0]
        days_since_service = features[1]
        critical_issues = features[4]

        urgency = 10  # Base urgency

        # Increase urgency based on factors
        urgency += (mileage / 120000) * 40
        urgency += (days_since_service / 90) * 30
        urgency += critical_issues * 20

        return min(100.0, urgency)
