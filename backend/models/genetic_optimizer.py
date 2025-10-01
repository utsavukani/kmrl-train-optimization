"""
KMRL Multi-Objective Genetic Algorithm Optimizer using DEAP
SIH25081 - NSGA-II implementation for train induction optimization

This module optimizes feasible solutions using genetic algorithms:
- Service readiness maximization
- Mileage balancing optimization  
- Branding exposure compliance
- Pareto-optimal solution selection
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    NSGA-II based multi-objective genetic algorithm for train optimization
    Optimizes service readiness, mileage balance, and branding compliance
    """

    def __init__(self):
        self.toolbox = None
        self.population_size = 50
        self.generations = 100
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1
        self.objectives = ['service_readiness',
                           'mileage_variance', 'branding_compliance']
        self.setup_deap_framework()

    def setup_deap_framework(self):
        """Initialize DEAP framework for multi-objective optimization"""

        # Create fitness and individual classes
        creator.create("FitnessMulti", base.Fitness,
                       weights=(1.0, -1.0, 1.0))  # max, min, max
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("evaluate", self.evaluate_objectives)

        logger.info(
            "DEAP framework initialized for multi-objective optimization")

    def optimize(self, feasible_space: List[Dict], objectives_config: Dict[str, float]) -> Dict:
        """
        Main optimization method using NSGA-II genetic algorithm

        Args:
            feasible_space: List of feasible train assignments from CP-SAT
            objectives_config: Weights and targets for optimization objectives

        Returns:
            Optimization result with best solutions and metrics
        """
        try:
            logger.info(
                f"Starting genetic algorithm optimization with {len(feasible_space)} feasible assignments")

            # Store feasible space and config for evaluation
            self.feasible_space = feasible_space
            self.objectives_config = objectives_config

            # Generate initial population
            population = self.generate_initial_population()

            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            logger.info(f"Initial population: {len(population)} individuals")

            # Evolution statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            # Run NSGA-II evolution (use DEAP implementation if available,
            # otherwise use a lightweight fallback implementation)
            final_population, logbook = self.run_nsga2_evolution(
                population, stats)

            # Extract Pareto front
            pareto_front = tools.sortNondominated(
                final_population, len(final_population))[0]

            # Select best solution based on objectives configuration
            best_solution = self.select_best_solution(pareto_front)

            # Generate optimization results
            optimization_result = self.generate_optimization_result(
                best_solution,
                pareto_front,
                logbook
            )

            logger.info(
                f"Optimization completed. Pareto front size: {len(pareto_front)}")
            return optimization_result

        except Exception as e:
            logger.error(f"Genetic algorithm optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'best_solution': [],
                'pareto_front': [],
                'metrics': {},
                'assignments': [],
                'objectives_achieved': {}
            }

    def generate_initial_population(self) -> List:
        """Generate initial population based on feasible assignments"""

        population = []

        for _ in range(self.population_size):
            # Create individual chromosome representing train assignments
            individual = []

            for train_assignment in self.feasible_space:
                # Chromosome encoding: [assignment_type, service_hours, bay_preference]
                chromosome = {
                    'train_id': train_assignment['train_id'],
                    'assignment': train_assignment['assignment'],
                    'service_hours': random.randint(6, 18) if train_assignment['assignment'] == 'SERVICE' else 0,
                    'bay_preference': random.randint(1, 12),
                    'priority_score': random.uniform(0.1, 1.0)
                }
                individual.append(chromosome)

            # Ensure constraints are maintained
            individual = self.repair_individual(individual)
            population.append(creator.Individual(individual))

        return population

    def custom_crossover(self, ind1, ind2):
        """Custom crossover operator preserving constraints"""

        # Create offspring
        offspring1 = creator.Individual(ind1[:])
        offspring2 = creator.Individual(ind2[:])

        # Perform uniform crossover on service hours and priorities
        for i in range(len(ind1)):
            if random.random() < 0.5:
                # Swap service hours
                offspring1[i]['service_hours'], offspring2[i]['service_hours'] = \
                    offspring2[i]['service_hours'], offspring1[i]['service_hours']

                # Swap priority scores
                offspring1[i]['priority_score'], offspring2[i]['priority_score'] = \
                    offspring2[i]['priority_score'], offspring1[i]['priority_score']

        # Repair to maintain constraints
        offspring1 = creator.Individual(self.repair_individual(offspring1))
        offspring2 = creator.Individual(self.repair_individual(offspring2))

        return offspring1, offspring2

    def custom_mutation(self, individual, indpb=0.1):
        """Custom mutation operator with constraint preservation"""

        for i, gene in enumerate(individual):
            if random.random() < indpb:
                # Mutate service hours (if in service)
                if gene['assignment'] == 'SERVICE':
                    gene['service_hours'] = max(6, min(18,
                                                       gene['service_hours'] + random.randint(-2, 2)))

                # Mutate bay preference
                gene['bay_preference'] = random.randint(1, 12)

                # Mutate priority score
                gene['priority_score'] = max(0.1, min(1.0,
                                                      gene['priority_score'] + random.uniform(-0.2, 0.2)))

        # Repair individual to maintain constraints
        individual = self.repair_individual(individual)
        return (creator.Individual(individual),)

    def repair_individual(self, individual: List[Dict]) -> List[Dict]:
        """Repair individual to maintain feasibility constraints"""

        # Ensure minimum service trains
        service_count = sum(
            1 for gene in individual if gene['assignment'] == 'SERVICE')
        min_service = self.objectives_config.get('min_service_trains', 18)

        if service_count < min_service:
            # Convert standby trains to service
            standby_trains = [
                gene for gene in individual if gene['assignment'] == 'STANDBY']
            needed = min_service - service_count

            for i in range(min(needed, len(standby_trains))):
                standby_trains[i]['assignment'] = 'SERVICE'
                standby_trains[i]['service_hours'] = random.randint(12, 18)

        # Ensure bay capacity constraints
        maintenance_trains = [
            gene for gene in individual if gene['assignment'] == 'MAINTENANCE']
        max_maintenance_bays = self.objectives_config.get(
            'maintenance_bays', 4)

        if len(maintenance_trains) > max_maintenance_bays:
            # Convert excess maintenance to standby
            excess = len(maintenance_trains) - max_maintenance_bays
            for i in range(excess):
                maintenance_trains[i]['assignment'] = 'STANDBY'
                maintenance_trains[i]['service_hours'] = 0

        return individual

    def evaluate_objectives(self, individual: List[Dict]) -> Tuple[float, float, float]:
        """Evaluate multi-objective fitness for an individual"""

        # Objective 1: Service Readiness (maximize)
        service_readiness = self.calculate_service_readiness(individual)

        # Objective 2: Mileage Variance (minimize)
        mileage_variance = self.calculate_mileage_variance(individual)

        # Objective 3: Branding Compliance (maximize)
        branding_compliance = self.calculate_branding_compliance(individual)

        return service_readiness, mileage_variance, branding_compliance

    def calculate_service_readiness(self, individual: List[Dict]) -> float:
        """Calculate service readiness score (0-100)"""

        service_trains = [
            gene for gene in individual if gene['assignment'] == 'SERVICE']
        total_service_hours = sum(gene['service_hours']
                                  for gene in service_trains)

        # Target: 18 trains × 16 hours = 288 service-hours
        target_service_hours = 288
        readiness_score = min(
            100, (total_service_hours / target_service_hours) * 100)

        # Penalty for under-utilization
        if len(service_trains) < 18:
            readiness_score *= 0.8

        return readiness_score

    def calculate_mileage_variance(self, individual: List[Dict]) -> float:
        """Calculate mileage distribution variance (minimize for balanced wear)"""

        service_trains = [
            gene for gene in individual if gene['assignment'] == 'SERVICE']

        if len(service_trains) == 0:
            return 100.0  # High penalty

        # Get historical mileage from feasible space
        mileages = []
        for gene in service_trains:
            train_data = next(
                (t for t in self.feasible_space if t['train_id'] == gene['train_id']), None)
            if train_data:
                historical_mileage = self.get_train_historical_mileage(
                    train_data['train_id'])
                projected_mileage = historical_mileage + \
                    (gene['service_hours'] * 25)  # 25 km/hour avg
                mileages.append(projected_mileage)

        # Calculate variance
        if len(mileages) > 1:
            variance = np.var(mileages)
            return variance / 1000  # Normalize to reasonable scale

        return 0.0

    def calculate_branding_compliance(self, individual: List[Dict]) -> float:
        """Calculate branding exposure compliance score (0-100)"""

        service_trains = [
            gene for gene in individual if gene['assignment'] == 'SERVICE']

        compliance_score = 0.0
        total_weight = 0.0

        for gene in service_trains:
            train_data = next(
                (t for t in self.feasible_space if t['train_id'] == gene['train_id']), None)
            if train_data:
                # Get branding requirements for this train
                required_exposure = self.get_branding_requirements(
                    train_data['train_id'])
                actual_exposure = gene['service_hours']

                # Calculate compliance ratio
                if required_exposure > 0:
                    compliance_ratio = min(
                        1.0, actual_exposure / required_exposure)
                    compliance_score += compliance_ratio * \
                        gene['priority_score']
                    total_weight += gene['priority_score']

        return (compliance_score / total_weight * 100) if total_weight > 0 else 0.0

    def select_best_solution(self, pareto_front: List) -> List[Dict]:
        """Select best solution from Pareto front based on objectives configuration"""

        if not pareto_front:
            return []

        # Calculate weighted scores for each solution
        best_solution = None
        best_score = -float('inf')

        weights = {
            'service_readiness': self.objectives_config.get('service_weight', 0.5),
            'mileage_balance': self.objectives_config.get('mileage_weight', 0.3),
            'branding_compliance': self.objectives_config.get('branding_weight', 0.2)
        }

        for solution in pareto_front:
            service_score, mileage_variance, branding_score = solution.fitness.values

            # Calculate weighted score (normalize mileage variance)
            # Convert variance to score
            normalized_mileage = max(0, 100 - mileage_variance)
            weighted_score = (
                weights['service_readiness'] * service_score +
                weights['mileage_balance'] * normalized_mileage +
                weights['branding_compliance'] * branding_score
            )

            if weighted_score > best_score:
                best_score = weighted_score
                best_solution = solution

        return best_solution if best_solution else pareto_front[0]

    def generate_optimization_result(self, best_solution: List[Dict], pareto_front: List, logbook) -> Dict:
        """Generate comprehensive optimization result"""

        # Convert best solution to assignment format
        assignments = []
        for gene in best_solution:
            assignments.append({
                'train_id': gene['train_id'],
                'assignment': gene['assignment'],
                'service_hours': gene['service_hours'],
                'bay_assignment': gene['bay_preference'],
                'priority_score': gene['priority_score'],
                'confidence': 0.85  # High confidence from CP-SAT + GA
            })

        # Calculate achieved objectives
        objectives_achieved = {
            'service_readiness': best_solution.fitness.values[0],
            'mileage_variance': best_solution.fitness.values[1],
            'branding_compliance': best_solution.fitness.values[2],
            'total_service_trains': len([a for a in assignments if a['assignment'] == 'SERVICE']),
            'total_service_hours': sum(a['service_hours'] for a in assignments)
        }

        # Extract evolution metrics
        metrics = {
            'generations_run': len(logbook),
            'pareto_front_size': len(pareto_front),
            'population_size': self.population_size,
            'convergence_data': [record['avg'] for record in logbook] if logbook else [],
            'final_diversity': self.calculate_diversity(pareto_front)
        }

        return {
            'success': True,
            'best_solution': best_solution,
            'assignments': assignments,
            'objectives_achieved': objectives_achieved,
            'pareto_front': pareto_front[:10],  # Top 10 solutions
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

    def run_nsga2_evolution(self, population, stats):
        """Run NSGA-II evolution using DEAP's algorithms.eaNSGA2 if present,
        otherwise run a simple generational NSGA-II-like loop as a fallback.
        Returns final population and a simple logbook list.
        """
        # Try DEAP built-in
        try:
            final_pop, logbook = algorithms.eaNSGA2(
                population,
                self.toolbox,
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob,
                ngen=self.generations,
                stats=stats,
                verbose=False
            )
            return final_pop, logbook
        except AttributeError:
            logger.warning(
                "DEAP.algorithms.eaNSGA2 not available, using fallback GA loop")

        # Fallback implementation: simple generational loop using selNSGA2
        logbook = []
        pop = population
        # Ensure fitness values exist for initial population
        for ind in pop:
            if not hasattr(ind, 'fitness') or not getattr(ind.fitness, 'values', None):
                ind.fitness.values = self.evaluate_objectives(ind)

        for gen in range(self.generations):
            # Create offspring (clone population)
            offspring = [creator.Individual(ind[:]) for ind in pop]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_prob:
                    c1, c2 = self.custom_crossover(
                        offspring[i-1], offspring[i])
                    offspring[i-1], offspring[i] = c1, c2

            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    mutated = self.custom_mutation(offspring[i])[0]
                    offspring[i] = mutated

            # Evaluate all offspring
            fitnesses = list(map(self.evaluate_objectives, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            # Combine and select next generation using NSGA-II selection
            combined = pop + offspring
            try:
                pop = tools.selNSGA2(combined, self.population_size)
            except Exception:
                # Fallback to simple selection by fitness sum if selNSGA2 fails
                combined.sort(key=lambda ind: sum(
                    ind.fitness.values), reverse=True)
                pop = combined[:self.population_size]

            # Record simple stats
            avg = np.mean([ind.fitness.values for ind in pop],
                          axis=0).tolist() if pop else []
            logbook.append({'gen': gen, 'avg': avg})

        return pop, logbook

    def calculate_diversity(self, pareto_front: List) -> float:
        """Calculate diversity metric for Pareto front"""

        if len(pareto_front) < 2:
            return 0.0

        # Calculate average distance between solutions
        distances = []
        for i in range(len(pareto_front)):
            for j in range(i + 1, len(pareto_front)):
                dist = np.linalg.norm(np.array(pareto_front[i].fitness.values) -
                                      np.array(pareto_front[j].fitness.values))
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def get_train_historical_mileage(self, train_id: str) -> int:
        """Get historical mileage for a train (mock implementation)"""
        # In production, this would query the actual database
        base_mileage = int(train_id.split('-')[-1]) * 10000  # Mock calculation
        return base_mileage + random.randint(5000, 15000)

    def get_branding_requirements(self, train_id: str) -> int:
        """Get branding exposure requirements for a train (mock implementation)"""
        # In production, this would query advertiser contracts
        brand_types = ['premium', 'standard', 'basic']
        brand_type = random.choice(brand_types)

        requirements = {
            'premium': 16,
            'standard': 12,
            'basic': 8
        }

        return requirements[brand_type]

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization configuration and performance"""

        return {
            'algorithm': 'NSGA-II Multi-Objective Genetic Algorithm',
            'population_size': self.population_size,
            'generations': self.generations,
            'crossover_probability': self.crossover_prob,
            'mutation_probability': self.mutation_prob,
            'objectives': self.objectives,
            'selection_method': 'NSGA-II Non-dominated Sorting',
            'crossover_method': 'Custom Constraint-Preserving Crossover',
            'mutation_method': 'Custom Constraint-Preserving Mutation'
        }
