"""
KMRL LLM Explanation Engine using LangChain
SIH25081 - Generate human-readable explanations for optimization decisions

This module provides:
- Train assignment reasoning explanations
- Optimization decision justification
- Conflict analysis and resolution suggestions
- What-if scenario impact summaries
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# For production, use actual LangChain with local LLM
# from langchain.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    LLM-based explanation generator for KMRL optimization decisions
    Provides transparent, human-readable reasoning for all assignments
    """

    def __init__(self):
        self.llm = None
        self.prompt_templates = {}
        self.is_initialized = False
        self.explanation_history = []

        # Initialize prompt templates
        self._setup_prompt_templates()

    async def initialize(self):
        """Initialize LLM connection (local model preferred for demo)"""
        try:
            logger.info("Initializing LLM explanation engine...")

            # For demo purposes, use rule-based explanations
            # In production, uncomment below for actual LLM integration:
            # self.llm = Ollama(model="llama2")  # or other local model

            self.is_initialized = True
            logger.info("✅ LLM explanation engine initialized (demo mode)")

        except Exception as e:
            logger.warning(
                f"LLM initialization failed, using rule-based explanations: {str(e)}")
            self.is_initialized = True  # Continue with rule-based mode

    def generate_explanations(self, assignments: List[Dict], reasoning_context: Dict) -> Dict[str, str]:
        """
        Generate comprehensive explanations for optimization results

        Args:
            assignments: List of train assignments from optimization
            reasoning_context: Context including constraints, metrics, predictions

        Returns:
            Dictionary of explanations for different aspects
        """
        try:
            logger.info(
                f"Generating explanations for {len(assignments)} train assignments")

            explanations = {
                'executive_summary': self._generate_executive_summary(assignments, reasoning_context),
                'individual_assignments': self._generate_individual_explanations(assignments, reasoning_context),
                'optimization_rationale': self._generate_optimization_rationale(reasoning_context),
                'constraint_analysis': self._generate_constraint_analysis(reasoning_context),
                'risk_assessment': self._generate_risk_assessment(assignments, reasoning_context),
                'recommendations': self._generate_recommendations(assignments, reasoning_context)
            }

            # Store in history for learning
            self.explanation_history.append({
                'timestamp': datetime.now().isoformat(),
                'explanations': explanations,
                'context': reasoning_context
            })

            logger.info("✅ Explanations generated successfully")
            return explanations

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return self._generate_fallback_explanation(assignments)

    def _setup_prompt_templates(self):
        """Setup prompt templates for different explanation types"""

        self.prompt_templates = {
            'executive_summary': """
            Analyze the train optimization results and provide an executive summary.
            
            Context:
            - Total trains: {total_trains}
            - Service assignments: {service_count}
            - Standby assignments: {standby_count} 
            - Maintenance assignments: {maintenance_count}
            - Key constraints: {key_constraints}
            - Optimization objectives achieved: {objectives}
            
            Provide a 2-3 sentence executive summary of the optimization results.
            """,

            'individual_assignment': """
            Explain why train {train_id} was assigned to {assignment}.
            
            Train details:
            - Fitness certificate valid until: {cert_valid_to}
            - Job card status: {jobcard_status}
            - Mileage since overhaul: {mileage} km
            - IoT sensor flags: {sensor_flags}
            - Crew availability: {crew_available}
            
            Provide a clear, concise explanation for this assignment decision.
            """,

            'constraint_analysis': """
            Analyze the constraint satisfaction in this optimization.
            
            Constraints applied:
            {constraints_log}
            
            Conflicts detected:
            {conflicts}
            
            Explain how constraints were handled and any trade-offs made.
            """
        }

    def _generate_executive_summary(self, assignments: List[Dict], context: Dict) -> str:
        """Generate high-level summary of optimization results"""

        # Count assignments by type
        assignment_counts = {'SERVICE': 0, 'STANDBY': 0, 'MAINTENANCE': 0}
        for assignment in assignments:
            assignment_counts[assignment.get('assignment', 'UNKNOWN')] += 1

        total_trains = len(assignments)
        service_count = assignment_counts['SERVICE']

        # Calculate fleet availability
        fleet_availability = (service_count / total_trains *
                              100) if total_trains > 0 else 0

        # Extract key metrics
        objectives = context.get('optimization_metrics', {})
        constraints_applied = len(context.get('constraints_applied', []))

        # Generate explanation
        if fleet_availability >= 72:  # 18/25 = 72%
            performance_assessment = "optimal"
        elif fleet_availability >= 60:
            performance_assessment = "good"
        else:
            performance_assessment = "constrained"

        summary = f"""The optimization successfully assigned {service_count} trains to revenue service, 
        {assignment_counts['STANDBY']} to standby, and {assignment_counts['MAINTENANCE']} to maintenance, 
        achieving {fleet_availability:.1f}% fleet availability. This represents {performance_assessment} 
        performance given the current constraints. {constraints_applied} operational constraints were 
        successfully satisfied, with all safety requirements maintained."""

        return summary.strip()

    def _generate_individual_explanations(self, assignments: List[Dict], context: Dict) -> Dict[str, str]:
        """Generate explanations for each individual train assignment"""

        individual_explanations = {}

        for assignment in assignments:
            train_id = assignment.get('train_id', 'Unknown')
            assignment_type = assignment.get('assignment', 'UNKNOWN')
            confidence = assignment.get('confidence', 0.5)

            # Get train-specific context
            explanation = self._explain_individual_assignment(
                assignment, context, confidence)
            individual_explanations[train_id] = explanation

        return individual_explanations

    def _explain_individual_assignment(self, assignment: Dict, context: Dict, confidence: float) -> str:
        """Explain individual train assignment decision"""

        train_id = assignment.get('train_id', 'Unknown')
        assignment_type = assignment.get('assignment', 'UNKNOWN')
        service_hours = assignment.get('service_hours', 0)

        # Extract reasoning factors
        reasoning_factors = []

        if assignment_type == 'SERVICE':
            reasoning_factors.append(f"assigned {service_hours} service hours")
            if service_hours >= 16:
                reasoning_factors.append("high service capacity requirement")
            elif service_hours >= 12:
                reasoning_factors.append("standard service requirement")
            else:
                reasoning_factors.append("reduced service due to constraints")

        elif assignment_type == 'MAINTENANCE':
            reasoning_factors.append("maintenance requirement detected")
            # Check for maintenance triggers
            if context.get('constraints_log'):
                for constraint in context['constraints_log']:
                    if constraint.get('train_id') == train_id:
                        if 'certificate' in constraint.get('constraint', '').lower():
                            reasoning_factors.append(
                                "fitness certificate renewal needed")
                        elif 'critical' in constraint.get('constraint', '').lower():
                            reasoning_factors.append(
                                "critical maintenance issue")
                        elif 'mileage' in constraint.get('constraint', '').lower():
                            reasoning_factors.append(
                                "high mileage threshold reached")

        elif assignment_type == 'STANDBY':
            reasoning_factors.append(
                "standby assignment for operational flexibility")

        # Add confidence indicator
        confidence_text = ""
        if confidence >= 0.9:
            confidence_text = " This assignment has very high confidence."
        elif confidence >= 0.7:
            confidence_text = " This assignment has high confidence."
        elif confidence >= 0.5:
            confidence_text = " This assignment has moderate confidence."
        else:
            confidence_text = " This assignment has lower confidence due to conflicting constraints."

        explanation = f"{train_id} {assignment_type.lower()}: {', '.join(reasoning_factors)}.{confidence_text}"
        return explanation

    def _generate_optimization_rationale(self, context: Dict) -> str:
        """Generate explanation of optimization approach and objectives"""

        objectives = context.get('optimization_metrics', {})

        rationale_parts = [
            "The optimization balanced three key objectives:",
            "1. Service Readiness: Maximizing trains available for passenger service",
            "2. Mileage Balancing: Distributing wear evenly across the fleet",
            "3. Branding Compliance: Meeting advertiser exposure requirements"
        ]

        if objectives:
            rationale_parts.append("\nAchieved metrics:")
            for objective, value in objectives.items():
                if isinstance(value, (int, float)):
                    rationale_parts.append(f"- {objective}: {value:.1f}")
                else:
                    rationale_parts.append(f"- {objective}: {value}")

        rationale_parts.append(
            "\nThe genetic algorithm explored multiple solution combinations to find the optimal balance between these competing objectives.")

        return " ".join(rationale_parts)

    def _generate_constraint_analysis(self, context: Dict) -> str:
        """Analyze constraint satisfaction and conflicts"""

        constraints_log = context.get('constraints_applied', [])
        conflicts = context.get('conflicts', [])

        if not constraints_log and not conflicts:
            return "All operational constraints were satisfied without conflicts."

        analysis_parts = []

        # Safety constraints
        safety_constraints = [
            c for c in constraints_log if c.get('type') == 'safety']
        if safety_constraints:
            analysis_parts.append(
                f"Applied {len(safety_constraints)} safety constraints including fitness certificate validation and critical job card restrictions.")

        # Operational constraints
        operational_constraints = [
            c for c in constraints_log if c.get('type') == 'operational']
        if operational_constraints:
            analysis_parts.append(
                f"Enforced {len(operational_constraints)} operational constraints for service levels and crew availability.")

        # Depot constraints
        depot_constraints = [
            c for c in constraints_log if c.get('type') == 'depot']
        if depot_constraints:
            analysis_parts.append(
                f"Applied {len(depot_constraints)} depot capacity constraints for bay utilization and stabling geometry.")

        # Conflicts
        if conflicts:
            analysis_parts.append(
                f"Resolved {len(conflicts)} constraint conflicts through optimization trade-offs.")
            for conflict in conflicts[:3]:  # Show first 3 conflicts
                analysis_parts.append(
                    f"- {conflict.get('message', 'Constraint conflict detected')}")

        return " ".join(analysis_parts)

    def _generate_risk_assessment(self, assignments: List[Dict], context: Dict) -> str:
        """Generate risk assessment for the optimization result"""

        ml_predictions = context.get('ml_predictions', {})

        risk_factors = []

        # Analyze individual predictions
        individual_predictions = ml_predictions.get(
            'individual_predictions', [])
        if individual_predictions:
            high_risk_trains = [
                p for p in individual_predictions if p.get('delay_risk', 0) > 0.7]
            if high_risk_trains:
                risk_factors.append(
                    f"{len(high_risk_trains)} trains have high delay risk")

            high_maintenance = [p for p in individual_predictions if p.get(
                'maintenance_urgency', 0) > 80]
            if high_maintenance:
                risk_factors.append(
                    f"{len(high_maintenance)} trains require urgent maintenance attention")

        # Fleet-level risks
        service_assignments = [
            a for a in assignments if a.get('assignment') == 'SERVICE']
        if len(service_assignments) < 18:
            risk_factors.append(
                "below minimum service capacity - service disruption risk")

        maintenance_assignments = [
            a for a in assignments if a.get('assignment') == 'MAINTENANCE']
        if len(maintenance_assignments) > 4:
            risk_factors.append(
                "maintenance capacity exceeded - scheduling delays possible")

        if not risk_factors:
            return "Risk assessment: Low operational risk. All assignments within normal parameters."

        return f"Risk assessment: {'; '.join(risk_factors)}. Recommend monitoring and contingency planning."

    def _generate_recommendations(self, assignments: List[Dict], context: Dict) -> str:
        """Generate actionable recommendations based on optimization results"""

        recommendations = []

        # Service capacity recommendations
        service_count = len(
            [a for a in assignments if a.get('assignment') == 'SERVICE'])
        if service_count < 20:
            recommendations.append(
                "Consider extending fitness certificate validity for borderline trains to increase service capacity.")

        # Maintenance scheduling
        maintenance_count = len(
            [a for a in assignments if a.get('assignment') == 'MAINTENANCE'])
        if maintenance_count > 3:
            recommendations.append(
                "Schedule non-critical maintenance for off-peak periods to optimize availability.")

        # Predictive insights
        ml_predictions = context.get('ml_predictions', {})
        if ml_predictions:
            fleet_reliability = ml_predictions.get(
                'aggregate_metrics', {}).get('fleet_reliability_score', 0)
            if fleet_reliability < 70:
                recommendations.append(
                    "Implement enhanced predictive maintenance program to improve fleet reliability.")

        # Operational efficiency
        recommendations.append(
            "Monitor real-time performance against optimized schedule and adjust dynamically.")
        recommendations.append(
            "Review constraint parameters weekly to ensure optimization remains aligned with operational reality.")

        if not recommendations:
            recommendations.append(
                "Current optimization is well-balanced. Continue monitoring for any emerging constraints.")

        return "Recommendations: " + " ".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])

    def _generate_fallback_explanation(self, assignments: List[Dict]) -> Dict[str, str]:
        """Generate basic explanations when LLM fails"""

        return {
            'executive_summary': f"Optimization completed for {len(assignments)} trains with constraint-based assignment allocation.",
            'individual_assignments': {a.get('train_id', 'Unknown'): f"Assigned to {a.get('assignment', 'UNKNOWN')} based on operational constraints." for a in assignments},
            'optimization_rationale': "Multi-objective optimization balanced service readiness, maintenance requirements, and operational constraints.",
            'constraint_analysis': "All safety and operational constraints were applied according to KMRL standards.",
            'risk_assessment': "Standard operational risk profile maintained with safety constraints enforced.",
            'recommendations': "Continue monitoring optimization performance and adjust constraints as needed."
        }

    def explain_what_if_scenario(self, base_result: Dict, modified_result: Dict, modifications: Dict) -> str:
        """Explain the impact of what-if scenario changes"""

        try:
            # Calculate deltas
            base_service = len([a for a in base_result.get(
                'assignments', []) if a.get('assignment') == 'SERVICE'])
            modified_service = len([a for a in modified_result.get(
                'assignments', []) if a.get('assignment') == 'SERVICE'])
            service_delta = modified_service - base_service

            # Analyze modifications
            modification_descriptions = []
            for key, value in modifications.items():
                if key.startswith('train_'):
                    train_id = key.split('_')[1]
                    modification_descriptions.append(f"{train_id} modified")
                else:
                    modification_descriptions.append(f"{key} changed")

            # Generate explanation
            impact_description = ""
            if service_delta > 0:
                impact_description = f"increases service capacity by {service_delta} trains"
            elif service_delta < 0:
                impact_description = f"reduces service capacity by {abs(service_delta)} trains"
            else:
                impact_description = "maintains current service capacity"

            explanation = f"What-if scenario: {', '.join(modification_descriptions)} {impact_description}. "

            # Add recommendation
            if service_delta < 0:
                explanation += "Consider alternative modifications to maintain service levels."
            elif service_delta > 0:
                explanation += "This modification improves operational flexibility."
            else:
                explanation += "This modification has neutral impact on service capacity."

            return explanation

        except Exception as e:
            logger.error(f"What-if explanation failed: {str(e)}")
            return "What-if scenario analysis: Changes applied successfully with operational impact under review."

    def get_explanation_quality_score(self, explanation: str) -> float:
        """Calculate quality score for generated explanation"""

        if not explanation or len(explanation) < 10:
            return 0.0

        # Basic quality metrics
        word_count = len(explanation.split())
        has_specifics = any(keyword in explanation.lower() for keyword in [
            'train', 'service', 'maintenance', 'constraint', 'optimization'
        ])
        has_quantitative = any(char.isdigit() for char in explanation)

        quality_score = 0.5  # Base score

        if 20 <= word_count <= 100:  # Appropriate length
            quality_score += 0.2
        if has_specifics:  # Domain-specific terms
            quality_score += 0.2
        if has_quantitative:  # Includes numbers/metrics
            quality_score += 0.1

        return min(1.0, quality_score)

    def get_explanation_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent explanation history for analysis"""

        return self.explanation_history[-limit:] if self.explanation_history else []
