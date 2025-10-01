"""
KMRL Data Validation Module
SIH25081 - Input data validation and cleaning utilities

This module provides:
- CSV data validation against schema
- JSON data structure validation  
- Data quality checks and cleaning
- Error reporting and suggestions
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for KMRL train induction system
    Validates input data quality and enforces business rules
    """

    def __init__(self):
        self.required_train_fields = [
            'train_id', 'last_service_date', 'fitness_cert_valid_from',
            'fitness_cert_valid_to', 'jobcard_status', 'mileage_since_overhaul'
        ]

        self.optional_fields = [
            'iot_sensor_flags', 'crew_available', 'branding_exposure_hours',
            'stabling_bay', 'cleaning_slot'
        ]

        self.validation_errors = []
        self.validation_warnings = []

    def validate_csv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate CSV data from file uploads (e.g., Maximo exports)

        Args:
            df: DataFrame containing train data

        Returns:
            Validation result with cleaned data and error summary
        """
        try:
            logger.info(f"Validating CSV data with {len(df)} records")

            self.validation_errors = []
            self.validation_warnings = []

            # Step 1: Structure validation
            structure_valid = self._validate_csv_structure(df)

            if not structure_valid:
                return {
                    'valid': False,
                    'errors': self.validation_errors,
                    'warnings': self.validation_warnings,
                    'cleaned_data': [],
                    'records_processed': 0
                }

            # Step 2: Field-by-field validation
            cleaned_records = []

            for index, row in df.iterrows():
                record_validation = self._validate_train_record(
                    row.to_dict(), index)

                if record_validation['valid']:
                    cleaned_records.append(record_validation['cleaned_record'])
                else:
                    self.validation_errors.extend(record_validation['errors'])

            # Step 3: Business rule validation
            business_validation = self._validate_business_rules(
                cleaned_records)

            # Step 4: Generate summary
            summary = self._generate_validation_summary(
                len(df), len(cleaned_records), len(
                    self.validation_errors), len(self.validation_warnings)
            )

            return {
                'valid': len(self.validation_errors) == 0,
                'cleaned_data': cleaned_records,
                'records_processed': len(df),
                'records_validated': len(cleaned_records),
                'summary': summary,
                'errors': self.validation_errors,
                'warnings': self.validation_warnings
            }

        except Exception as e:
            logger.error(f"CSV validation failed: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'cleaned_data': [],
                'records_processed': 0
            }

    def validate_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate JSON data from API requests or IoT sensors

        Args:
            data: Dictionary containing train or system data

        Returns:
            Validation result with cleaned data
        """
        try:
            logger.info("Validating JSON data")

            self.validation_errors = []
            self.validation_warnings = []

            if isinstance(data, dict) and 'trains' in data:
                # Multiple trains data
                cleaned_trains = []

                for i, train_data in enumerate(data['trains']):
                    record_validation = self._validate_train_record(
                        train_data, i)

                    if record_validation['valid']:
                        cleaned_trains.append(
                            record_validation['cleaned_record'])
                    else:
                        self.validation_errors.extend(
                            record_validation['errors'])

                return {
                    'valid': len(self.validation_errors) == 0,
                    'cleaned_data': cleaned_trains,
                    'records_processed': len(data['trains']),
                    'errors': self.validation_errors,
                    'warnings': self.validation_warnings
                }

            elif isinstance(data, dict) and 'train_id' in data:
                # Single train data
                record_validation = self._validate_train_record(data, 0)

                return {
                    'valid': record_validation['valid'],
                    'cleaned_data': [record_validation['cleaned_record']] if record_validation['valid'] else [],
                    'records_processed': 1,
                    'errors': record_validation['errors'],
                    'warnings': self.validation_warnings
                }

            else:
                return {
                    'valid': False,
                    'error': 'Invalid JSON structure. Expected train data.',
                    'cleaned_data': []
                }

        except Exception as e:
            logger.error(f"JSON validation failed: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'cleaned_data': []
            }

    def _validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """Validate CSV file structure and required columns"""

        # Check if DataFrame is empty
        if df.empty:
            self.validation_errors.append({
                'type': 'structure',
                'message': 'CSV file is empty',
                'severity': 'critical'
            })
            return False

        # Check required columns
        missing_columns = []
        for field in self.required_train_fields:
            if field not in df.columns:
                missing_columns.append(field)

        if missing_columns:
            self.validation_errors.append({
                'type': 'structure',
                'message': f'Missing required columns: {", ".join(missing_columns)}',
                'severity': 'critical'
            })
            return False

        # Check for duplicate train IDs
        if 'train_id' in df.columns:
            duplicates = df[df.duplicated(
                'train_id', keep=False)]['train_id'].tolist()
            if duplicates:
                self.validation_warnings.append({
                    'type': 'structure',
                    'message': f'Duplicate train IDs found: {list(set(duplicates))}',
                    'severity': 'warning'
                })

        return True

    def _validate_train_record(self, record: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        """Validate individual train record"""

        errors = []
        warnings = []
        cleaned_record = record.copy()

        # Validate train ID
        train_id_validation = self._validate_train_id(
            record.get('train_id'), row_index)
        if train_id_validation['errors']:
            errors.extend(train_id_validation['errors'])
        if train_id_validation['cleaned_value']:
            cleaned_record['train_id'] = train_id_validation['cleaned_value']

        # Validate dates
        date_fields = ['last_service_date',
                       'fitness_cert_valid_from', 'fitness_cert_valid_to']
        for field in date_fields:
            if field in record:
                date_validation = self._validate_date(
                    record.get(field), field, row_index)
                if date_validation['errors']:
                    errors.extend(date_validation['errors'])
                if date_validation['warnings']:
                    warnings.extend(date_validation['warnings'])
                if date_validation['cleaned_value']:
                    cleaned_record[field] = date_validation['cleaned_value']

        # Validate mileage
        if 'mileage_since_overhaul' in record:
            mileage_validation = self._validate_mileage(
                record.get('mileage_since_overhaul'), row_index)
            if mileage_validation['errors']:
                errors.extend(mileage_validation['errors'])
            if mileage_validation['warnings']:
                warnings.extend(mileage_validation['warnings'])
            if mileage_validation['cleaned_value'] is not None:
                cleaned_record['mileage_since_overhaul'] = mileage_validation['cleaned_value']

        # Validate job card status
        if 'jobcard_status' in record:
            jobcard_validation = self._validate_jobcard_status(
                record.get('jobcard_status'), row_index)
            if jobcard_validation['warnings']:
                warnings.extend(jobcard_validation['warnings'])
            if jobcard_validation['cleaned_value']:
                cleaned_record['jobcard_status'] = jobcard_validation['cleaned_value']

        # Validate IoT sensor flags
        if 'iot_sensor_flags' in record:
            iot_validation = self._validate_iot_flags(
                record.get('iot_sensor_flags'), row_index)
            if iot_validation['warnings']:
                warnings.extend(iot_validation['warnings'])
            if iot_validation['cleaned_value']:
                cleaned_record['iot_sensor_flags'] = iot_validation['cleaned_value']

        # Validate crew availability
        if 'crew_available' in record:
            crew_validation = self._validate_boolean_field(
                record.get('crew_available'), 'crew_available', row_index)
            if crew_validation['cleaned_value'] is not None:
                cleaned_record['crew_available'] = crew_validation['cleaned_value']

        # Validate branding exposure hours
        if 'branding_exposure_hours' in record:
            branding_validation = self._validate_numeric_field(
                record.get(
                    'branding_exposure_hours'), 'branding_exposure_hours', 0, 24, row_index
            )
            if branding_validation['errors']:
                errors.extend(branding_validation['errors'])
            if branding_validation['cleaned_value'] is not None:
                cleaned_record['branding_exposure_hours'] = branding_validation['cleaned_value']

        return {
            'valid': len(errors) == 0,
            'cleaned_record': cleaned_record,
            'errors': errors,
            'warnings': warnings
        }

    def _validate_train_id(self, train_id: Any, row_index: int) -> Dict[str, Any]:
        """Validate train ID format and uniqueness"""

        errors = []
        cleaned_value = None

        if not train_id:
            errors.append({
                'type': 'validation',
                'field': 'train_id',
                'row': row_index,
                'message': 'Train ID is required',
                'severity': 'critical'
            })
            return {'errors': errors, 'cleaned_value': cleaned_value}

        # Clean and validate train ID format
        train_id_str = str(train_id).strip().upper()

        # Expected format: KMRL-XXX (where XXX is 3-digit number)
        if not re.match(r'^KMRL-\d{3}$', train_id_str):
            errors.append({
                'type': 'validation',
                'field': 'train_id',
                'row': row_index,
                'message': f'Invalid train ID format: {train_id_str}. Expected format: KMRL-001',
                'severity': 'critical'
            })
        else:
            cleaned_value = train_id_str

        return {'errors': errors, 'cleaned_value': cleaned_value}

    def _validate_date(self, date_value: Any, field_name: str, row_index: int) -> Dict[str, Any]:
        """Validate date fields"""

        errors = []
        warnings = []
        cleaned_value = None

        if not date_value:
            errors.append({
                'type': 'validation',
                'field': field_name,
                'row': row_index,
                'message': f'{field_name} is required',
                'severity': 'critical'
            })
            return {'errors': errors, 'warnings': warnings, 'cleaned_value': cleaned_value}

        # Try to parse date
        try:
            if isinstance(date_value, str):
                # Try multiple date formats
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']
                parsed_date = None

                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(
                            date_value.strip(), fmt)
                        break
                    except ValueError:
                        continue

                if not parsed_date:
                    errors.append({
                        'type': 'validation',
                        'field': field_name,
                        'row': row_index,
                        'message': f'Invalid date format: {date_value}. Use YYYY-MM-DD format',
                        'severity': 'critical'
                    })
                    return {'errors': errors, 'warnings': warnings, 'cleaned_value': cleaned_value}

                cleaned_value = parsed_date.strftime('%Y-%m-%d')

                # Business logic validation
                if field_name == 'fitness_cert_valid_to':
                    # Check if certificate is expired
                    if parsed_date < datetime.now():
                        warnings.append({
                            'type': 'business_rule',
                            'field': field_name,
                            'row': row_index,
                            'message': f'Fitness certificate expired on {cleaned_value}',
                            'severity': 'warning'
                        })

                elif field_name == 'last_service_date':
                    # Check if service is overdue (more than 30 days)
                    days_since_service = (datetime.now() - parsed_date).days
                    if days_since_service > 30:
                        warnings.append({
                            'type': 'business_rule',
                            'field': field_name,
                            'row': row_index,
                            'message': f'Service overdue by {days_since_service - 30} days',
                            'severity': 'warning'
                        })

        except Exception as e:
            errors.append({
                'type': 'validation',
                'field': field_name,
                'row': row_index,
                'message': f'Date parsing error: {str(e)}',
                'severity': 'critical'
            })

        return {'errors': errors, 'warnings': warnings, 'cleaned_value': cleaned_value}

    def _validate_mileage(self, mileage: Any, row_index: int) -> Dict[str, Any]:
        """Validate mileage field"""

        errors = []
        warnings = []
        cleaned_value = None

        try:
            if mileage is None or mileage == '':
                errors.append({
                    'type': 'validation',
                    'field': 'mileage_since_overhaul',
                    'row': row_index,
                    'message': 'Mileage since overhaul is required',
                    'severity': 'critical'
                })
                return {'errors': errors, 'warnings': warnings, 'cleaned_value': cleaned_value}

            # Convert to integer
            mileage_int = int(float(str(mileage).replace(',', '')))

            # Validate range
            if mileage_int < 0:
                errors.append({
                    'type': 'validation',
                    'field': 'mileage_since_overhaul',
                    'row': row_index,
                    'message': 'Mileage cannot be negative',
                    'severity': 'critical'
                })
            elif mileage_int > 150000:  # Reasonable upper limit
                warnings.append({
                    'type': 'validation',
                    'field': 'mileage_since_overhaul',
                    'row': row_index,
                    'message': f'Very high mileage: {mileage_int} km. Verify accuracy.',
                    'severity': 'warning'
                })
                cleaned_value = mileage_int
            elif mileage_int > 100000:  # Overhaul needed
                warnings.append({
                    'type': 'business_rule',
                    'field': 'mileage_since_overhaul',
                    'row': row_index,
                    'message': f'Overhaul due - mileage: {mileage_int} km',
                    'severity': 'warning'
                })
                cleaned_value = mileage_int
            else:
                cleaned_value = mileage_int

        except (ValueError, TypeError):
            errors.append({
                'type': 'validation',
                'field': 'mileage_since_overhaul',
                'row': row_index,
                'message': f'Invalid mileage value: {mileage}. Must be a number.',
                'severity': 'critical'
            })

        return {'errors': errors, 'warnings': warnings, 'cleaned_value': cleaned_value}

    def _validate_jobcard_status(self, status: Any, row_index: int) -> Dict[str, Any]:
        """Validate job card status"""

        warnings = []
        cleaned_value = None

        if not status:
            cleaned_value = 'NONE'
        else:
            status_str = str(status).strip().upper()

            # Valid statuses
            valid_statuses = ['NONE', 'OPEN',
                              'CLOSED', 'CRITICAL_OPEN', 'PENDING']

            if status_str not in valid_statuses:
                warnings.append({
                    'type': 'validation',
                    'field': 'jobcard_status',
                    'row': row_index,
                    'message': f'Unknown job card status: {status_str}. Using as-is.',
                    'severity': 'warning'
                })

            if 'CRITICAL' in status_str:
                warnings.append({
                    'type': 'business_rule',
                    'field': 'jobcard_status',
                    'row': row_index,
                    'message': 'Critical job card detected - service assignment restricted',
                    'severity': 'warning'
                })

            cleaned_value = status_str

        return {'warnings': warnings, 'cleaned_value': cleaned_value}

    def _validate_iot_flags(self, flags: Any, row_index: int) -> Dict[str, Any]:
        """Validate IoT sensor flags"""

        warnings = []
        cleaned_value = None

        if not flags:
            cleaned_value = 'NORMAL'
        else:
            flags_str = str(flags).strip().upper()

            if 'CRITICAL' in flags_str:
                warnings.append({
                    'type': 'business_rule',
                    'field': 'iot_sensor_flags',
                    'row': row_index,
                    'message': 'Critical IoT sensor alert detected',
                    'severity': 'warning'
                })

            cleaned_value = flags_str

        return {'warnings': warnings, 'cleaned_value': cleaned_value}

    def _validate_boolean_field(self, value: Any, field_name: str, row_index: int) -> Dict[str, Any]:
        """Validate boolean fields"""

        cleaned_value = None

        if value is None or value == '':
            cleaned_value = True  # Default to True
        else:
            value_str = str(value).strip().lower()

            if value_str in ['true', '1', 'yes', 'y']:
                cleaned_value = True
            elif value_str in ['false', '0', 'no', 'n']:
                cleaned_value = False
            else:
                cleaned_value = bool(value)

        return {'cleaned_value': cleaned_value}

    def _validate_numeric_field(self, value: Any, field_name: str, min_val: float,
                                max_val: float, row_index: int) -> Dict[str, Any]:
        """Validate numeric fields with range checks"""

        errors = []
        cleaned_value = None

        if value is None or value == '':
            cleaned_value = 0.0
        else:
            try:
                num_value = float(str(value).replace(',', ''))

                if num_value < min_val or num_value > max_val:
                    errors.append({
                        'type': 'validation',
                        'field': field_name,
                        'row': row_index,
                        'message': f'{field_name} must be between {min_val} and {max_val}',
                        'severity': 'critical'
                    })
                else:
                    cleaned_value = num_value

            except (ValueError, TypeError):
                errors.append({
                    'type': 'validation',
                    'field': field_name,
                    'row': row_index,
                    'message': f'Invalid numeric value for {field_name}: {value}',
                    'severity': 'critical'
                })

        return {'errors': errors, 'cleaned_value': cleaned_value}

    def _validate_business_rules(self, records: List[Dict]) -> Dict[str, Any]:
        """Validate business rules across all records"""

        if not records:
            return {'valid': True, 'violations': []}

        violations = []

        # Rule 1: Maximum 25 trains (KMRL fleet size)
        if len(records) > 25:
            violations.append({
                'type': 'business_rule',
                'rule': 'fleet_size_limit',
                'message': f'Fleet size exceeds limit: {len(records)} trains (max: 25)',
                'severity': 'warning'
            })

        # Rule 2: Check for service capacity
        service_capable = sum(
            1 for record in records if self._is_service_capable(record))
        if service_capable < 18:  # Minimum service requirement
            violations.append({
                'type': 'business_rule',
                'rule': 'minimum_service_capacity',
                'message': f'Only {service_capable} trains capable of service (minimum: 18)',
                'severity': 'critical'
            })

        return {'valid': len(violations) == 0, 'violations': violations}

    def _is_service_capable(self, record: Dict) -> bool:
        """Check if train is capable of service assignment"""

        # Check fitness certificate
        try:
            cert_valid_to = datetime.strptime(record.get(
                'fitness_cert_valid_to', ''), '%Y-%m-%d')
            if cert_valid_to < datetime.now():
                return False
        except:
            return False

        # Check critical job cards
        if 'CRITICAL' in record.get('jobcard_status', ''):
            return False

        # Check IoT alerts
        if 'CRITICAL' in record.get('iot_sensor_flags', ''):
            return False

        return True

    def _generate_validation_summary(self, total_records: int, valid_records: int,
                                     error_count: int, warning_count: int) -> Dict[str, Any]:
        """Generate validation summary report"""

        success_rate = (valid_records / total_records *
                        100) if total_records > 0 else 0

        return {
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': total_records - valid_records,
            'success_rate': round(success_rate, 1),
            'error_count': error_count,
            'warning_count': warning_count,
            'validation_status': 'success' if error_count == 0 else 'failed',
            'timestamp': datetime.now().isoformat()
        }
