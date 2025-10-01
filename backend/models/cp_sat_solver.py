"""
KMRL Constraint Programming Solver using Google OR-Tools CP-SAT
SIH25081 - Constraint satisfaction for train induction planning

This module implements hard constraint filtering using CP-SAT:
- Safety constraints (fitness certificates, job cards)
- Operational constraints (bay capacity, crew availability)
- Regulatory compliance constraints
"""

from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class ConstraintEngine:
    """
    CP-SAT based constraint programming engine for train scheduling
    Filters feasible assignments before genetic algorithm optimization
    """

    def __init__(self):
        self.model = None
        self.solver = None
        self.variables = {}
        self.constraints_log = []

    def filter_feasible_assignments(self, trains_data: List[Dict], constraints: Dict[str, Any]) -> Dict:
        """
        Main method to filter feasible train assignments using CP-SAT
        Returns: feasible solution space or conflicts if infeasible
        """
        try:
            logger.info(
                f"Filtering feasible assignments for {len(trains_data)} trains")

            # Initialize CP-SAT model
            self.model = cp_model.CpModel()
            self.variables = {}
            self.constraints_log = []

            # Create decision variables for each train assignment
            self._create_decision_variables(trains_data)

            # Add safety constraints (hard constraints)
            self._add_safety_constraints(trains_data, constraints)

            # Add operational constraints
            self._add_operational_constraints(trains_data, constraints)

            # Add depot capacity constraints
            self._add_depot_constraints(trains_data, constraints)

            # Solve the constraint satisfaction problem
            solution_result = self._solve_constraints()

            if solution_result['feasible']:
                # Generate feasible solution space for genetic algorithm
                feasible_space = self._generate_solution_space(
                    trains_data, solution_result)
                return {
                    'feasible': True,
                    'solution_space': feasible_space,
                    'constraints_log': self.constraints_log,
                    'solver_stats': solution_result['stats']
                }
            else:
                # Analyze conflicts and suggest remediation
                conflicts = self._analyze_conflicts(trains_data, constraints)
                return {
                    'feasible': False,
                    'conflicts': conflicts,
                    'constraints_log': self.constraints_log,
                    'remediation_suggestions': self._suggest_remediation(conflicts)
                }

        except Exception as e:
            logger.error(f"Constraint filtering failed: {str(e)}")
            return {
                'feasible': False,
                'error': str(e),
                'conflicts': [{'type': 'system_error', 'message': str(e)}]
            }

    def _create_decision_variables(self, trains_data: List[Dict]):
        """Create CP-SAT decision variables for train assignments"""

        # Assignment variables: train_assignment[train_id] = 0(service), 1(standby), 2(maintenance)
        for train in trains_data:
            train_id = train['train_id']
            self.variables[f'assignment_{train_id}'] = self.model.NewIntVar(
                0, 2, f'assignment_{train_id}'
            )

            # Bay assignment variables
            self.variables[f'bay_{train_id}'] = self.model.NewIntVar(
                1, 12, f'bay_{train_id}'  # Assuming 12 bays at KMRL depot
            )

            # Service hour variables (for branding compliance)
            self.variables[f'service_hours_{train_id}'] = self.model.NewIntVar(
                # 18-hour operational window
                0, 18, f'service_hours_{train_id}'
            )

        logger.info(f"Created {len(self.variables)} decision variables")

    def _add_safety_constraints(self, trains_data: List[Dict], constraints: Dict):
        """Add hard safety constraints that cannot be violated"""

        safety_violations = 0

        for train in trains_data:
            train_id = train['train_id']
            assignment_var = self.variables[f'assignment_{train_id}']

            # Constraint 1: Fitness certificate validity
            cert_valid_from = datetime.strptime(
                train['fitness_cert_valid_from'], '%Y-%m-%d')
            cert_valid_to = datetime.strptime(
                train['fitness_cert_valid_to'], '%Y-%m-%d')
            today = datetime.now()

            if today < cert_valid_from or today > cert_valid_to:
                # Force to maintenance if certificate expired/not valid
                self.model.Add(assignment_var == 2)
                self.constraints_log.append({
                    'type': 'safety',
                    'train_id': train_id,
                    'constraint': 'fitness_certificate_invalid',
                    'action': 'forced_maintenance'
                })
                safety_violations += 1

            # Constraint 2: Critical job cards
            if train['jobcard_status'] == 'CRITICAL_OPEN':
                # Cannot assign to service if critical job cards are open
                self.model.Add(assignment_var != 0)
                self.constraints_log.append({
                    'type': 'safety',
                    'train_id': train_id,
                    'constraint': 'critical_jobcard_open',
                    'action': 'service_prohibited'
                })
                safety_violations += 1

            # Constraint 3: IoT sensor alerts
            if 'CRITICAL' in train.get('iot_sensor_flags', ''):
                # Force immediate maintenance for critical sensor alerts
                self.model.Add(assignment_var == 2)
                self.constraints_log.append({
                    'type': 'safety',
                    'train_id': train_id,
                    'constraint': 'critical_iot_alert',
                    'action': 'forced_maintenance'
                })
                safety_violations += 1

        logger.info(
            f"Added safety constraints with {safety_violations} violations detected")

    def _add_operational_constraints(self, trains_data: List[Dict], constraints: Dict):
        """Add operational constraints for service requirements"""

        # Constraint 1: Minimum trains in service
        min_service_trains = constraints.get('min_service_trains', 18)
        service_trains = []

        for train in trains_data:
            train_id = train['train_id']
            assignment_var = self.variables[f'assignment_{train_id}']

            # Create boolean variable for service assignment
            is_service = self.model.NewBoolVar(f'is_service_{train_id}')
            self.model.Add(assignment_var == 0).OnlyEnforceIf(is_service)
            self.model.Add(assignment_var != 0).OnlyEnforceIf(is_service.Not())
            service_trains.append(is_service)

        # Ensure minimum service trains
        self.model.Add(sum(service_trains) >= min_service_trains)
        self.constraints_log.append({
            'type': 'operational',
            'constraint': 'minimum_service_trains',
            'requirement': min_service_trains
        })

        # Constraint 2: Crew availability
        available_crews = constraints.get('available_crews', 22)
        crew_required = []

        for train in trains_data:
            train_id = train['train_id']
            assignment_var = self.variables[f'assignment_{train_id}']

            if not train.get('crew_available', True):
                # Cannot assign to service without crew
                self.model.Add(assignment_var != 0)
                self.constraints_log.append({
                    'type': 'operational',
                    'train_id': train_id,
                    'constraint': 'crew_unavailable',
                    'action': 'service_prohibited'
                })
            else:
                # Count crew requirements for service trains
                needs_crew = self.model.NewBoolVar(f'needs_crew_{train_id}')
                self.model.Add(assignment_var == 0).OnlyEnforceIf(needs_crew)
                self.model.Add(assignment_var != 0).OnlyEnforceIf(
                    needs_crew.Not())
                crew_required.append(needs_crew)

        # Ensure crew capacity not exceeded
        self.model.Add(sum(crew_required) <= available_crews)
        self.constraints_log.append({
            'type': 'operational',
            'constraint': 'crew_capacity',
            'available': available_crews
        })

        logger.info(
            "Added operational constraints for service requirements and crew")

    def _add_depot_constraints(self, trains_data: List[Dict], constraints: Dict):
        """Add depot capacity and bay assignment constraints"""

        # Constraint 1: Bay capacity limits
        maintenance_bays = constraints.get('maintenance_bays', 4)
        cleaning_bays = constraints.get('cleaning_bays', 3)

        maintenance_trains = []
        cleaning_trains = []

        for train in trains_data:
            train_id = train['train_id']
            assignment_var = self.variables[f'assignment_{train_id}']

            # Count trains assigned to maintenance
            in_maintenance = self.model.NewBoolVar(
                f'in_maintenance_{train_id}')
            self.model.Add(assignment_var == 2).OnlyEnforceIf(in_maintenance)
            self.model.Add(assignment_var != 2).OnlyEnforceIf(
                in_maintenance.Not())
            maintenance_trains.append(in_maintenance)

            # Count trains needing cleaning
            if train.get('cleaning_slot') == 'REQUIRED':
                needs_cleaning = self.model.NewBoolVar(
                    f'needs_cleaning_{train_id}')
                cleaning_trains.append(needs_cleaning)

        # Enforce bay capacity limits
        self.model.Add(sum(maintenance_trains) <= maintenance_bays)
        if cleaning_trains:
            self.model.Add(sum(cleaning_trains) <= cleaning_bays)

        self.constraints_log.append({
            'type': 'depot',
            'constraint': 'bay_capacity',
            'maintenance_limit': maintenance_bays,
            'cleaning_limit': cleaning_bays
        })

        # Constraint 2: Stabling geometry optimization
        self._add_stabling_constraints(trains_data)

        logger.info("Added depot capacity and stabling constraints")

    def _add_stabling_constraints(self, trains_data: List[Dict]):
        """Add constraints to minimize shunting operations"""

        # Adjacent bay assignment preference for service trains
        for i, train1 in enumerate(trains_data):
            for j, train2 in enumerate(trains_data):
                if i >= j:
                    continue

                train1_id = train1['train_id']
                train2_id = train2['train_id']

                bay1 = self.variables[f'bay_{train1_id}']
                bay2 = self.variables[f'bay_{train2_id}']
                assignment1 = self.variables[f'assignment_{train1_id}']
                assignment2 = self.variables[f'assignment_{train2_id}']

                # If both trains are in service, prefer adjacent bays
                both_service = self.model.NewBoolVar(
                    f'both_service_{train1_id}_{train2_id}')
                self.model.Add(assignment1 == 0).OnlyEnforceIf(both_service)
                self.model.Add(assignment2 == 0).OnlyEnforceIf(both_service)

                # Minimize bay distance (soft constraint, will be handled in objective)

    def _solve_constraints(self) -> Dict:
        """Solve the CP-SAT model and return results"""

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 30.0  # 30-second timeout

        logger.info("Solving constraint satisfaction problem...")
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(
                f"Solution found with status: {self.solver.StatusName(status)}")
            return {
                'feasible': True,
                'status': self.solver.StatusName(status),
                'stats': {
                    'solve_time': self.solver.WallTime(),
                    'objective_value': self.solver.ObjectiveValue() if status == cp_model.OPTIMAL else None,
                    'num_branches': self.solver.NumBranches(),
                    'num_conflicts': self.solver.NumConflicts()
                }
            }
        else:
            logger.warning(
                f"No feasible solution found: {self.solver.StatusName(status)}")
            return {
                'feasible': False,
                'status': self.solver.StatusName(status),
                'stats': {
                    'solve_time': self.solver.WallTime(),
                    'num_branches': self.solver.NumBranches(),
                    'num_conflicts': self.solver.NumConflicts()
                }
            }

    def _generate_solution_space(self, trains_data: List[Dict], solution_result: Dict) -> List[Dict]:
        """Generate feasible solution space for genetic algorithm"""

        feasible_assignments = []

        for train in trains_data:
            train_id = train['train_id']
            assignment_var = self.variables[f'assignment_{train_id}']
            bay_var = self.variables[f'bay_{train_id}']
            service_hours_var = self.variables[f'service_hours_{train_id}']

            assignment_value = self.solver.Value(assignment_var)
            bay_value = self.solver.Value(bay_var)
            service_hours_value = self.solver.Value(service_hours_var)

            assignment_map = {0: 'SERVICE', 1: 'STANDBY', 2: 'MAINTENANCE'}

            feasible_assignments.append({
                'train_id': train_id,
                'assignment': assignment_map[assignment_value],
                'bay': bay_value,
                'service_hours': service_hours_value,
                'feasible': True,
                'constraints_satisfied': True
            })

        logger.info(
            f"Generated feasible solution space with {len(feasible_assignments)} assignments")
        return feasible_assignments

    def _analyze_conflicts(self, trains_data: List[Dict], constraints: Dict) -> List[Dict]:
        """Analyze constraint conflicts when no feasible solution exists"""

        conflicts = []

        # Check for over-constrained scenarios
        service_capable_trains = sum(1 for train in trains_data
                                     if self._is_service_capable(train))
        min_required = constraints.get('min_service_trains', 18)

        if service_capable_trains < min_required:
            conflicts.append({
                'type': 'capacity_shortage',
                'message': f'Only {service_capable_trains} trains available for service, need {min_required}',
                'severity': 'critical',
                'affected_trains': service_capable_trains
            })

        # Check maintenance bay overflow
        maintenance_required = sum(1 for train in trains_data
                                   if self._requires_maintenance(train))
        maintenance_bays = constraints.get('maintenance_bays', 4)

        if maintenance_required > maintenance_bays:
            conflicts.append({
                'type': 'maintenance_overflow',
                'message': f'{maintenance_required} trains need maintenance, only {maintenance_bays} bays available',
                'severity': 'high',
                'overflow_count': maintenance_required - maintenance_bays
            })

        return conflicts

    def _suggest_remediation(self, conflicts: List[Dict]) -> List[Dict]:
        """Suggest remediation actions for conflicts"""

        suggestions = []

        for conflict in conflicts:
            if conflict['type'] == 'capacity_shortage':
                suggestions.append({
                    'conflict_type': 'capacity_shortage',
                    'actions': [
                        'Extend fitness certificate validity for borderline trains',
                        'Defer non-critical maintenance to next cycle',
                        'Request additional crew assignments',
                        'Consider reduced service frequency'
                    ],
                    'priority': 'urgent'
                })

            elif conflict['type'] == 'maintenance_overflow':
                suggestions.append({
                    'conflict_type': 'maintenance_overflow',
                    'actions': [
                        'Prioritize critical safety maintenance only',
                        'Schedule overflow maintenance for next day',
                        'Request additional maintenance bay allocation',
                        'Consider external maintenance facility'
                    ],
                    'priority': 'high'
                })

        return suggestions

    def _is_service_capable(self, train: Dict) -> bool:
        """Check if train is capable of service assignment"""

        # Check fitness certificate
        cert_valid_to = datetime.strptime(
            train['fitness_cert_valid_to'], '%Y-%m-%d')
        if datetime.now() > cert_valid_to:
            return False

        # Check critical job cards
        if train['jobcard_status'] == 'CRITICAL_OPEN':
            return False

        # Check IoT sensor alerts
        if 'CRITICAL' in train.get('iot_sensor_flags', ''):
            return False

        return True

    def _requires_maintenance(self, train: Dict) -> bool:
        """Check if train requires immediate maintenance"""

        return (not self._is_service_capable(train) or
                train.get('mileage_since_overhaul', 0) > 100000 or
                'MAINTENANCE_DUE' in train.get('iot_sensor_flags', ''))

    def get_constraint_summary(self) -> Dict:
        """Get summary of applied constraints"""

        summary = {
            'total_constraints': len(self.constraints_log),
            'safety_constraints': len([c for c in self.constraints_log if c['type'] == 'safety']),
            'operational_constraints': len([c for c in self.constraints_log if c['type'] == 'operational']),
            'depot_constraints': len([c for c in self.constraints_log if c['type'] == 'depot']),
            'constraints_log': self.constraints_log
        }

        return summary
