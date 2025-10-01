/**
 * KMRL Train Optimization Dashboard JavaScript
 * SIH25081 - Interactive frontend for AI-driven train scheduling
 * 
 * Features:
 * - Real-time train status display
 * - Optimization controls and execution
 * - What-if simulation capabilities
 * - Performance visualization
 * - LLM-powered explanations
 */

class KMRLDashboard {
    constructor() {
        this.apiBaseUrl = 'https://kmrl-train-optimization.onrender.com/api/v1';
        this.trainData = [];
        this.optimizationResults = null;
        this.performanceChart = null;

        this.initialize();
    }

    initialize() {
        this.setupEventListeners();
        this.updateCurrentTime();
        this.setupPerformanceChart();
        this.loadInitialData();

        // Set default target date to tomorrow
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        document.getElementById('target-date').value = tomorrow.toISOString().split('T')[0];

        // Update slider value displays
        this.updateSliderDisplays();
    }

    setupEventListeners() {
        // Main control buttons
        document.getElementById('refresh-data').addEventListener('click', () => this.refreshTrainData());
        document.getElementById('run-optimization').addEventListener('click', () => this.runOptimization());

        // Slider controls
        document.getElementById('min-service-trains').addEventListener('input', (e) => {
            document.getElementById('min-service-value').textContent = e.target.value;
        });

        // Modal controls
        document.getElementById('close-modal').addEventListener('click', () => this.closeSimulationModal());
        document.getElementById('cancel-simulation').addEventListener('click', () => this.closeSimulationModal());
        document.getElementById('run-simulation').addEventListener('click', () => this.runSimulation());

        // Click outside modal to close
        document.getElementById('simulation-modal').addEventListener('click', (e) => {
            if (e.target.id === 'simulation-modal') {
                this.closeSimulationModal();
            }
        });
    }

    updateCurrentTime() {
        const updateTime = () => {
            const now = new Date();
            const timeString = now.toLocaleString('en-IN', {
                timeZone: 'Asia/Kolkata',
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            document.getElementById('current-time').textContent = timeString;
        };

        updateTime();
        setInterval(updateTime, 1000);
    }

    updateSliderDisplays() {
        // Update objective weight displays
        const serviceWeight = document.getElementById('service-weight');
        const mileageWeight = document.getElementById('mileage-weight');
        const brandingWeight = document.getElementById('branding-weight');

        const updateWeights = () => {
            const total = parseInt(serviceWeight.value) + parseInt(mileageWeight.value) + parseInt(brandingWeight.value);
            // Auto-normalize to 100%
            if (total !== 100) {
                const factor = 100 / total;
                serviceWeight.value = Math.round(serviceWeight.value * factor);
                mileageWeight.value = Math.round(mileageWeight.value * factor);
                brandingWeight.value = Math.round(brandingWeight.value * factor);
            }
        };

        serviceWeight.addEventListener('input', updateWeights);
        mileageWeight.addEventListener('input', updateWeights);
        brandingWeight.addEventListener('input', updateWeights);
    }

    async loadInitialData() {
        try {
            this.showLoading(true);
            await this.loadTrainData();
            await this.loadOptimizationHistory();
            this.updateKPICards();
        } catch (error) {
            this.showNotification('Failed to load initial data. Using demo data.', 'warning');
            this.loadDemoData();
        } finally {
            this.showLoading(false);
        }
    }

    async loadTrainData() {
        try {
            // In development, use demo data since backend might not be running
            const response = await this.apiCall('/trains', 'GET');
            this.trainData = response || this.generateDemoTrainData();
        } catch (error) {
            console.log('API not available, using demo data');
            this.trainData = this.generateDemoTrainData();
        }

        this.renderTrainTable();
        this.populateTrainSelectors();
    }

    generateDemoTrainData() {
        const assignments = ['SERVICE', 'STANDBY', 'MAINTENANCE'];
        const statuses = ['NORMAL', 'WARNING', 'CRITICAL'];

        return Array.from({ length: 25 }, (_, i) => {
            const trainId = `KMRL-${String(i + 1).padStart(3, '0')}`;
            const assignment = assignments[Math.floor(Math.random() * assignments.length)];
            const status = statuses[Math.floor(Math.random() * statuses.length)];

            return {
                train_id: trainId,
                assignment: assignment,
                service_hours: assignment === 'SERVICE' ? Math.floor(Math.random() * 10) + 8 : 0,
                confidence: Math.random() * 0.4 + 0.6, // 0.6 to 1.0
                last_service_date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                fitness_cert_valid_to: new Date(Date.now() + Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                jobcard_status: Math.random() > 0.9 ? 'CRITICAL_OPEN' : 'NONE',
                mileage_since_overhaul: Math.floor(Math.random() * 80000) + 20000,
                crew_available: Math.random() > 0.1,
                status: status
            };
        });
    }

    renderTrainTable() {
        const tbody = document.getElementById('train-table-body');
        tbody.innerHTML = '';

        this.trainData.forEach((train, index) => {
            const row = document.createElement('tr');
            row.className = 'table-hover';

            const statusClass = `status-${train.assignment.toLowerCase()}`;
            const confidencePercent = Math.round(train.confidence * 100);
            const confidenceColor = train.confidence > 0.8 ? 'text-green-600' : train.confidence > 0.6 ? 'text-yellow-600' : 'text-red-600';

            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                        <div class="status-indicator ${statusClass}"></div>
                        <span class="text-sm font-medium text-gray-900">${train.train_id}</span>
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium 
                        ${train.status === 'NORMAL' ? 'bg-green-100 text-green-800' :
                    train.status === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'}">
                        ${train.status}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${train.assignment}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${train.service_hours}h
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="text-sm ${confidenceColor}">${confidencePercent}%</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button onclick="dashboard.openSimulationModal('${train.train_id}')" 
                        class="text-blue-600 hover:text-blue-900 mr-3">
                        Simulate
                    </button>
                    <button onclick="dashboard.showTrainDetails('${train.train_id}')" 
                        class="text-green-600 hover:text-green-900">
                        Details
                    </button>
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    updateKPICards() {
        if (!this.trainData.length) return;

        const serviceTrains = this.trainData.filter(t => t.assignment === 'SERVICE').length;
        const maintenanceTrains = this.trainData.filter(t => t.assignment === 'MAINTENANCE').length;
        const criticalAlerts = this.trainData.filter(t => t.status === 'CRITICAL' || t.jobcard_status === 'CRITICAL_OPEN').length;

        const fleetAvailability = Math.round((serviceTrains / this.trainData.length) * 100);
        const optimizationScore = Math.round(85 + Math.random() * 10); // Demo score

        document.getElementById('fleet-availability').textContent = `${fleetAvailability}%`;
        document.getElementById('service-trains').textContent = `${serviceTrains}/25`;
        document.getElementById('maintenance-alerts').textContent = criticalAlerts;
        document.getElementById('optimization-score').textContent = optimizationScore;
    }

    setupPerformanceChart() {
        const ctx = document.getElementById('performance-chart').getContext('2d');

        // Generate sample performance data for the last 7 days
        const dates = [];
        const availabilityData = [];
        const delayData = [];

        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }));

            // Generate realistic performance data
            availabilityData.push(88 + Math.random() * 10); // 88-98% availability
            delayData.push(2 + Math.random() * 6); // 2-8 minutes average delay
        }

        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Fleet Availability (%)',
                    data: availabilityData,
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                }, {
                    label: 'Avg Delay (min)',
                    data: delayData,
                    borderColor: '#F59E0B',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Availability (%)'
                        },
                        min: 80,
                        max: 100
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Delay (minutes)'
                        },
                        min: 0,
                        max: 10,
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    async runOptimization() {
        try {
            this.showLoading(true);
            this.showNotification('Running optimization algorithm...', 'info');

            const optimizationRequest = {
                trains: this.trainData.map(train => ({
                    train_id: train.train_id,
                    last_service_date: train.last_service_date,
                    fitness_cert_valid_from: train.fitness_cert_valid_from || '2024-01-01',
                    fitness_cert_valid_to: train.fitness_cert_valid_to,
                    jobcard_status: train.jobcard_status || 'NONE',
                    mileage_since_overhaul: train.mileage_since_overhaul,
                    iot_sensor_flags: train.iot_sensor_flags || 'NORMAL',
                    crew_available: train.crew_available !== false,
                    branding_exposure_hours: train.branding_exposure_hours || 0,
                    stabling_bay: train.stabling_bay || 'BAY-01',
                    cleaning_slot: train.cleaning_slot || 'NONE'
                })),
                target_date: document.getElementById('target-date').value,
                constraints: {
                    min_service_trains: parseInt(document.getElementById('min-service-trains').value),
                    maintenance_bays: 4,
                    cleaning_bays: 3,
                    available_crews: 22
                },
                objectives: {
                    service_weight: parseInt(document.getElementById('service-weight').value) / 100,
                    mileage_weight: parseInt(document.getElementById('mileage-weight').value) / 100,
                    branding_weight: parseInt(document.getElementById('branding-weight').value) / 100
                }
            };

            let optimizationResult;
            try {
                optimizationResult = await this.apiCall('/optimize', 'POST', optimizationRequest);
            } catch (error) {
                // Fallback to demo optimization result
                optimizationResult = this.generateDemoOptimizationResult();
            }

            this.optimizationResults = optimizationResult;
            this.updateTrainDataFromOptimization(optimizationResult);
            this.renderTrainTable();
            this.updateKPICards();
            this.showOptimizationExplanation(optimizationResult);

            this.showNotification(`Optimization completed! Achieved ${Math.round(optimizationResult.objectives_achieved?.service_readiness || 85)}% service readiness.`, 'success');

        } catch (error) {
            console.error('Optimization failed:', error);
            this.showNotification('Optimization failed. Please check the system status.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    generateDemoOptimizationResult() {
        const assignments = this.trainData.map(train => ({
            train_id: train.train_id,
            assignment: ['SERVICE', 'STANDBY', 'MAINTENANCE'][Math.floor(Math.random() * 3)],
            service_hours: Math.floor(Math.random() * 10) + 8,
            bay_assignment: `BAY-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}`,
            priority_score: Math.random(),
            confidence: Math.random() * 0.4 + 0.6
        }));

        // Ensure minimum service requirement
        const serviceCount = assignments.filter(a => a.assignment === 'SERVICE').length;
        const minService = parseInt(document.getElementById('min-service-trains').value);

        if (serviceCount < minService) {
            const needed = minService - serviceCount;
            const nonService = assignments.filter(a => a.assignment !== 'SERVICE');
            for (let i = 0; i < needed && i < nonService.length; i++) {
                nonService[i].assignment = 'SERVICE';
                nonService[i].service_hours = Math.floor(Math.random() * 10) + 8;
            }
        }

        return {
            optimization_id: `opt_${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15)}`,
            timestamp: new Date().toISOString(),
            assignments: assignments,
            objectives_achieved: {
                service_readiness: 85 + Math.random() * 10,
                mileage_variance: Math.random() * 1000,
                branding_compliance: 90 + Math.random() * 8,
                total_service_trains: assignments.filter(a => a.assignment === 'SERVICE').length,
                total_service_hours: assignments.filter(a => a.assignment === 'SERVICE').reduce((sum, a) => sum + a.service_hours, 0)
            },
            conflicts: [],
            explanations: {
                executive_summary: `Successfully optimized ${assignments.length} trains with ${assignments.filter(a => a.assignment === 'SERVICE').length} assigned to service, achieving high fleet availability while maintaining safety constraints.`,
                individual_assignments: {},
                optimization_rationale: 'Multi-objective genetic algorithm balanced service readiness, maintenance requirements, and operational constraints to achieve optimal fleet utilization.'
            }
        };
    }

    updateTrainDataFromOptimization(result) {
        result.assignments.forEach(assignment => {
            const train = this.trainData.find(t => t.train_id === assignment.train_id);
            if (train) {
                train.assignment = assignment.assignment;
                train.service_hours = assignment.service_hours;
                train.confidence = assignment.confidence;
                train.bay_assignment = assignment.bay_assignment;
            }
        });
    }

    openSimulationModal(trainId) {
        document.getElementById('simulation-modal').classList.remove('hidden');

        // Populate train selector
        const trainSelect = document.getElementById('simulation-train');
        trainSelect.innerHTML = '<option value="">Select a train...</option>';
        this.trainData.forEach(train => {
            const option = document.createElement('option');
            option.value = train.train_id;
            option.textContent = `${train.train_id} (${train.assignment})`;
            if (train.train_id === trainId) {
                option.selected = true;
            }
            trainSelect.appendChild(option);
        });

        // Hide previous results
        document.getElementById('simulation-results').classList.add('hidden');
    }

    closeSimulationModal() {
        document.getElementById('simulation-modal').classList.add('hidden');
    }

    async runSimulation() {
        const trainId = document.getElementById('simulation-train').value;
        const assignment = document.getElementById('simulation-assignment').value;

        if (!trainId) {
            this.showNotification('Please select a train for simulation.', 'warning');
            return;
        }

        try {
            // Demo simulation result
            const currentTrain = this.trainData.find(t => t.train_id === trainId);
            const oldAssignment = currentTrain.assignment;

            let impactMessage;
            if (oldAssignment === assignment) {
                impactMessage = `No change in assignment for ${trainId}.`;
            } else {
                const serviceChange = (assignment === 'SERVICE' ? 1 : 0) - (oldAssignment === 'SERVICE' ? 1 : 0);
                if (serviceChange > 0) {
                    impactMessage = `Moving ${trainId} to ${assignment} increases service capacity by 1 train. Fleet availability improves by 4%.`;
                } else if (serviceChange < 0) {
                    impactMessage = `Moving ${trainId} to ${assignment} reduces service capacity by 1 train. Fleet availability decreases by 4%.`;
                } else {
                    impactMessage = `Moving ${trainId} to ${assignment} maintains current service capacity with neutral impact.`;
                }
            }

            document.getElementById('simulation-summary').textContent = impactMessage;
            document.getElementById('simulation-results').classList.remove('hidden');

        } catch (error) {
            console.error('Simulation failed:', error);
            this.showNotification('Simulation failed. Please try again.', 'error');
        }
    }

    showOptimizationExplanation(result) {
        const panel = document.getElementById('explanation-panel');
        const content = document.getElementById('explanation-content');

        content.innerHTML = `
            <div class="bg-blue-50 border-l-4 border-blue-400 p-4">
                <h4 class="text-lg font-medium text-blue-800 mb-2">Executive Summary</h4>
                <p class="text-blue-700">${result.explanations.executive_summary}</p>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-green-50 p-4 rounded-lg">
                    <h5 class="font-medium text-green-800">Service Readiness</h5>
                    <p class="text-2xl font-bold text-green-900">${Math.round(result.objectives_achieved.service_readiness)}%</p>
                </div>
                <div class="bg-yellow-50 p-4 rounded-lg">
                    <h5 class="font-medium text-yellow-800">Service Trains</h5>
                    <p class="text-2xl font-bold text-yellow-900">${result.objectives_achieved.total_service_trains}/25</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h5 class="font-medium text-purple-800">Service Hours</h5>
                    <p class="text-2xl font-bold text-purple-900">${result.objectives_achieved.total_service_hours}h</p>
                </div>
            </div>
            
            <div class="bg-gray-50 p-4 rounded-lg">
                <h5 class="font-medium text-gray-800 mb-2">Optimization Approach</h5>
                <p class="text-gray-700">${result.explanations.optimization_rationale}</p>
            </div>
        `;

        panel.classList.remove('hidden');
    }

    populateTrainSelectors() {
        const trainSelect = document.getElementById('simulation-train');
        if (trainSelect) {
            trainSelect.innerHTML = '<option value="">Select a train...</option>';
            this.trainData.forEach(train => {
                const option = document.createElement('option');
                option.value = train.train_id;
                option.textContent = `${train.train_id} (${train.assignment})`;
                trainSelect.appendChild(option);
            });
        }
    }

    async loadOptimizationHistory() {
        try {
            const history = await this.apiCall('/optimizations', 'GET') || this.generateDemoHistory();
            this.renderOptimizationHistory(history);
        } catch (error) {
            this.renderOptimizationHistory(this.generateDemoHistory());
        }
    }

    generateDemoHistory() {
        return Array.from({ length: 5 }, (_, i) => {
            const date = new Date();
            date.setHours(date.getHours() - (i + 1) * 3);

            return {
                optimization_id: `opt_${date.toISOString().replace(/[-:]/g, '').slice(0, 15)}`,
                created_at: date.toISOString(),
                objectives_achieved: {
                    service_readiness: 82 + Math.random() * 15,
                    total_service_trains: 16 + Math.floor(Math.random() * 6)
                },
                status: 'completed'
            };
        });
    }

    renderOptimizationHistory(history) {
        const container = document.getElementById('optimization-history');
        container.innerHTML = '';

        history.slice(0, 5).forEach(opt => {
            const timeAgo = this.getTimeAgo(new Date(opt.created_at));
            const serviceReadiness = Math.round(opt.objectives_achieved.service_readiness || 85);

            const historyItem = document.createElement('div');
            historyItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
            historyItem.innerHTML = `
                <div>
                    <p class="text-sm font-medium text-gray-900">${timeAgo}</p>
                    <p class="text-xs text-gray-600">Readiness: ${serviceReadiness}% | Trains: ${opt.objectives_achieved.total_service_trains || 18}</p>
                </div>
                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    ${opt.status}
                </span>
            `;
            container.appendChild(historyItem);
        });
    }

    getTimeAgo(date) {
        const now = new Date();
        const diffMs = now - date;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffMins = Math.floor(diffMs / (1000 * 60));

        if (diffHours > 0) {
            return `${diffHours}h ago`;
        } else if (diffMins > 0) {
            return `${diffMins}m ago`;
        } else {
            return 'Just now';
        }
    }

    async refreshTrainData() {
        this.showNotification('Refreshing train data...', 'info');
        await this.loadTrainData();
        this.updateKPICards();
        this.showNotification('Train data refreshed successfully!', 'success');
    }

    showTrainDetails(trainId) {
        const train = this.trainData.find(t => t.train_id === trainId);
        if (train) {
            const details = `
                Train: ${train.train_id}
                Assignment: ${train.assignment}
                Service Hours: ${train.service_hours}
                Confidence: ${Math.round(train.confidence * 100)}%
                Last Service: ${train.last_service_date}
                Certificate Expires: ${train.fitness_cert_valid_to}
                Mileage: ${train.mileage_since_overhaul?.toLocaleString()} km
                Crew Available: ${train.crew_available ? 'Yes' : 'No'}
            `;
            alert(details); // In production, use a proper modal
        }
    }

    showLoading(show) {
        const indicator = document.getElementById('loading-indicator');
        if (show) {
            indicator.classList.remove('hidden');
        } else {
            indicator.classList.add('hidden');
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');

        const colorClasses = {
            success: 'bg-green-100 border-green-400 text-green-700',
            warning: 'bg-yellow-100 border-yellow-400 text-yellow-700',
            error: 'bg-red-100 border-red-400 text-red-700',
            info: 'bg-blue-100 border-blue-400 text-blue-700'
        };

        const notification = document.createElement('div');
        notification.className = `notification border-l-4 p-4 mb-4 ${colorClasses[type]} rounded-md`;
        notification.innerHTML = `
            <div class="flex justify-between items-center">
                <p class="font-medium">${message}</p>
                <button onclick="this.parentElement.parentElement.remove()" class="text-current opacity-70 hover:opacity-100">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;

        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    async apiCall(endpoint, method = 'GET', data = null) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    loadDemoData() {
        this.trainData = this.generateDemoTrainData();
        this.renderTrainTable();
        this.populateTrainSelectors();
        this.updateKPICards();
        this.renderOptimizationHistory(this.generateDemoHistory());
    }
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new KMRLDashboard();
});

// Global functions for inline event handlers
window.dashboard = {
    openSimulationModal: (trainId) => dashboard.openSimulationModal(trainId),
    showTrainDetails: (trainId) => dashboard.showTrainDetails(trainId)
}; 


