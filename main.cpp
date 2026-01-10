#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// ---------- VECTOR OPERATIONS ----------

struct vec2 {
    float x, y;
    vec2 operator+(const vec2& other) const { return { x + other.x, y + other.y }; }
    vec2 operator-(const vec2& other) const { return { x - other.x, y - other.y }; }
    vec2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
    vec2& operator+=(const vec2& other) { x += other.x; y += other.y; return *this; }
    vec2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
};

float length(const vec2& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

vec2 normalize(const vec2& v) {
    float len = length(v);
    if (len < 1e-8f) return { 0.0f, 0.0f };
    return { v.x / len, v.y / len };
}

// ---------- PHYSICAL CONSTANTS ----------

const float BOLTZMANN_EV = 8.617e-5f;  // eV/K
const float ROOM_TEMP = 300.0f;         // Kelvin

// Silicon properties at 300K
const float ELECTRON_MOBILITY_300K = 1350.0f;  // cm²/(V·s) - for display only
const float HOLE_MOBILITY_300K = 480.0f;       // cm²/(V·s) - for display only
const float MOBILITY_TEMP_EXPONENT_E = 2.4f;
const float MOBILITY_TEMP_EXPONENT_H = 2.2f;

// Simulation-space constants (normalized units for visualization)
const float SIM_ELECTRON_MOBILITY = 0.008f;  // Simulation units
const float SIM_HOLE_MOBILITY = 0.003f;      // Holes are slower

const float ELECTRON_VSAT_NORM = 0.5f;  // Normalized saturation velocity
const float HOLE_VSAT_NORM = 0.4f;

// Device dimensions in simulation space
const float DEVICE_LENGTH = 1.8f;  // From -0.9 to 0.9

// Generation/Recombination rates - tuned for visual stability
const float GENERATION_BASE = 0.0000005f;    // Base thermal generation rate (lowered more)
const float GENERATION_TEMP_SCALE = 50.0f;   // Temperature sensitivity (steeper curve)

// Recombination: uses Shockley-Read-Hall inspired model
const float CARRIER_LIFETIME_BASE = 0.01f;   // Base recombination probability (increased)
const float AUGER_COEFFICIENT = 0.000005f;   // High-injection Auger-like term (increased)
const float RECOMBINATION_RADIUS = 0.15f;    // How close carriers must be to recombine

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// Simulation state
float appliedVoltage = 0.0f;
float temperature = 0.0f;  // Start at room temperature (300K)
float simulationSpeed = 1.0f;

// Current measurement
float currentAccumulator = 0.0f;
float currentMetric = 0.0f;
float measureTimer = 0.0f;
const float MEASURE_INTERVAL = 30.0f;

// ---------- PARTICLE STRUCTURE ----------

struct Particle {
    vec2 position;
    vec2 velocity;
    vec2 homePosition;
    bool isFree;
    bool isElectron;
    bool markedForDeletion;
    int partnerId;  // For tracking recombination partners

    Particle() : position({ 0,0 }), velocity({ 0,0 }), homePosition({ 0,0 }),
        isFree(false), isElectron(true), markedForDeletion(false), partnerId(-1) {
    }
};

// ---------- HELPER FUNCTIONS ----------

// Temperature-dependent mobility (normalized for simulation)
float calculateMobility(bool isElectron, float tempKelvin) {
    float baseMobility = isElectron ? SIM_ELECTRON_MOBILITY : SIM_HOLE_MOBILITY;
    float exponent = isElectron ? MOBILITY_TEMP_EXPONENT_E : MOBILITY_TEMP_EXPONENT_H;
    return baseMobility * pow(ROOM_TEMP / tempKelvin, exponent);
}

// Physical mobility for display
float calculatePhysicalMobility(bool isElectron, float tempKelvin) {
    float baseMobility = isElectron ? ELECTRON_MOBILITY_300K : HOLE_MOBILITY_300K;
    float exponent = isElectron ? MOBILITY_TEMP_EXPONENT_E : MOBILITY_TEMP_EXPONENT_H;
    return baseMobility * pow(ROOM_TEMP / tempKelvin, exponent);
}

// Calculate local carrier density gradient for diffusion
vec2 calculateDensityGradient(vec2 position, const vector<Particle>& particles, bool forElectrons) {
    const float SAMPLE_RADIUS = 0.2f;

    float leftDensity = 0.0f, rightDensity = 0.0f;
    float upDensity = 0.0f, downDensity = 0.0f;

    for (const Particle& p : particles) {
        if (!p.isFree || p.isElectron != forElectrons || p.markedForDeletion) continue;

        vec2 diff = p.position - position;
        float dist = length(diff);

        if (dist < SAMPLE_RADIUS && dist > 0.01f) {
            // Weight by inverse distance for smoother gradient
            float weight = 1.0f - (dist / SAMPLE_RADIUS);

            if (diff.x > 0) rightDensity += weight;
            else leftDensity += weight;

            if (diff.y > 0) upDensity += weight;
            else downDensity += weight;
        }
    }

    // Gradient points from low to high concentration
    return { rightDensity - leftDensity, upDensity - downDensity };
}

// Find nearest carrier of opposite type for recombination
int findRecombinationPartner(int myIndex, const vector<Particle>& particles) {
    const Particle& me = particles[myIndex];
    if (!me.isFree || me.markedForDeletion) return -1;

    float minDist = RECOMBINATION_RADIUS;
    int bestPartner = -1;

    for (size_t i = 0; i < particles.size(); i++) {
        if (i == myIndex) continue;
        const Particle& other = particles[i];

        // Must be opposite type, free, and not already marked
        if (!other.isFree || other.markedForDeletion) continue;
        if (other.isElectron == me.isElectron) continue;

        float dist = length(other.position - me.position);
        if (dist < minDist) {
            minDist = dist;
            bestPartner = (int)i;
        }
    }

    return bestPartner;
}

// ---------- SHADERS ----------

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aIsFree;
layout (location = 2) in float aIsElectron;
out float vIsFree;
out float vIsElectron;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    vIsFree = aIsFree;
    vIsElectron = aIsElectron;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in float vIsFree;
in float vIsElectron;
out vec4 FragColor;
void main() {
    if (vIsFree > 0.5) {
        if (vIsElectron > 0.5)
            FragColor = vec4(0.2, 0.8, 1.0, 1.0);  // Cyan for free electrons
        else
            FragColor = vec4(1.0, 0.3, 0.3, 1.0);  // Red for holes
    } else {
        FragColor = vec4(0.25, 0.3, 0.4, 1.0);    // Dark blue-gray for bound electrons
    }
}
)";

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// ---------- MAIN ----------

int main()
{
    srand(static_cast<unsigned int>(time(0)));

    // GLFW initialization
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Semiconductor Physics Simulation", NULL, NULL);
    if (window == NULL) {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check vertex shader compilation
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        cout << "Vertex shader compilation failed: " << infoLog << endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        cout << "Fragment shader compilation failed: " << infoLog << endl;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        cout << "Shader program linking failed: " << infoLog << endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create initial crystal lattice of bound electrons
    vector<Particle> particles;
    const int LATTICE_ROWS = 25;
    const int LATTICE_COLS = 40;
    particles.reserve(6000);  // Room for lattice + free carriers

    for (int col = 0; col < LATTICE_COLS; col++) {
        for (int row = 0; row < LATTICE_ROWS; row++) {
            Particle p;
            p.position.x = (float)col / (LATTICE_COLS - 1) * 1.7f - 0.85f;
            p.position.y = (float)row / (LATTICE_ROWS - 1) * 1.5f - 0.75f;
            p.homePosition = p.position;
            p.velocity = { 0.0f, 0.0f };
            p.isFree = false;
            p.isElectron = true;
            p.markedForDeletion = false;
            p.partnerId = -1;
            particles.push_back(p);
        }
    }

    const int LATTICE_SIZE = LATTICE_ROWS * LATTICE_COLS;

    // GPU buffers
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, 6000 * 5 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // IsFree attribute
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // IsElectron attribute
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glEnable(GL_PROGRAM_POINT_SIZE);

    // Main simulation loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Control panel
        ImGui::Begin("Semiconductor Lab Controls");

        ImGui::Text("Device Parameters");
        ImGui::SliderFloat("Applied Voltage (V)", &appliedVoltage, -5.0f, 10.0f);
        ImGui::SliderFloat("Temperature Factor", &temperature, 0.0f, 5.0f);
        ImGui::SliderFloat("Simulation Speed", &simulationSpeed, 0.1f, 3.0f);

        if (ImGui::Button("Reset Carriers")) {
            // Remove all free carriers, reset lattice
            particles.erase(
                remove_if(particles.begin(), particles.end(),
                    [](const Particle& p) { return p.isFree; }),
                particles.end()
            );
            currentAccumulator = 0.0f;
            currentMetric = 0.0f;
        }

        ImGui::Separator();
        ImGui::Text("Physical Properties");

        float T_kelvin = ROOM_TEMP + temperature * 50.0f;
        float kT = T_kelvin * BOLTZMANN_EV;

        ImGui::Text("Temperature: %.0f K (%.1f C)", T_kelvin, T_kelvin - 273.15f);
        ImGui::Text("Thermal Energy kT: %.4f eV", kT);
        ImGui::Text("Electric Field: %.2f V/cm", appliedVoltage / (DEVICE_LENGTH * 1e-4f));

        float mu_e = calculatePhysicalMobility(true, T_kelvin);
        float mu_h = calculatePhysicalMobility(false, T_kelvin);
        ImGui::Text("Electron Mobility: %.0f cm2/(V.s)", mu_e);
        ImGui::Text("Hole Mobility: %.0f cm2/(V.s)", mu_h);

        ImGui::Separator();
        ImGui::Text("Carrier Statistics");

        int freeElectrons = 0, holes = 0, boundElectrons = 0;
        for (const Particle& p : particles) {
            if (p.markedForDeletion) continue;
            if (p.isFree) {
                if (p.isElectron) freeElectrons++;
                else holes++;
            }
            else {
                boundElectrons++;
            }
        }

        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Free Electrons: %d", freeElectrons);
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Holes: %d", holes);
        ImGui::Text("Bound Electrons: %d", boundElectrons);
        ImGui::Text("Total Particles: %d", (int)particles.size());

        // Charge neutrality check
        int netCharge = holes - freeElectrons;  // Holes are +, free electrons are -
        if (abs(netCharge) > 2) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Net Charge: %+d (imbalanced!)", netCharge);
        }
        else {
            ImGui::Text("Net Charge: %+d (balanced)", netCharge);
        }

        ImGui::Separator();
        ImGui::Text("Current Flow");

        if (abs(currentMetric) > 0.1f) {
            ImVec4 color = currentMetric > 0 ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(1.0f, 0.5f, 0.2f, 1.0f);
            ImGui::TextColored(color, "Current: %.2f uA", currentMetric);
        }
        else {
            ImGui::Text("Current: %.2f uA", currentMetric);
        }

        ImGui::Separator();
        ImGui::Text("Performance: %.1f FPS", ImGui::GetIO().Framerate);

        ImGui::End();

        // ========== PHYSICS SIMULATION ==========

        float dt = 0.016f * simulationSpeed;  // ~60fps timestep
        float T_k = ROOM_TEMP + temperature * 50.0f;
        float kT_sim = T_k * BOLTZMANN_EV;

        // Electric field = Voltage / Length (in simulation units)
        float electricField = appliedVoltage / DEVICE_LENGTH;

        // Count current carriers
        int electronCount = 0, holeCount = 0;
        for (const Particle& p : particles) {
            if (p.isFree && !p.markedForDeletion) {
                if (p.isElectron) electronCount++;
                else holeCount++;
            }
        }

        // ===== GENERATION: Thermal excitation of bound electrons =====
        // Only bound electrons in the lattice can be excited
        float generationProb = GENERATION_BASE * exp((T_k - ROOM_TEMP) / GENERATION_TEMP_SCALE);

        // Limit total free carriers to prevent runaway
        bool canGenerate = (particles.size() < 5500) && (electronCount + holeCount < 400);

        for (size_t i = 0; i < particles.size() && i < (size_t)LATTICE_SIZE; i++) {
            Particle& p = particles[i];

            if (!p.isFree && p.isElectron && canGenerate) {
                float r = (float)rand() / RAND_MAX;

                if (r < generationProb * dt) {
                    // Excite this electron - it becomes free
                    p.isFree = true;
                    p.velocity = { 0.0f, 0.0f };

                    // Create corresponding hole at same location
                    Particle hole;
                    hole.position = p.position;
                    hole.homePosition = p.homePosition;
                    hole.velocity = { 0.0f, 0.0f };
                    hole.isFree = true;
                    hole.isElectron = false;
                    hole.markedForDeletion = false;
                    hole.partnerId = -1;
                    particles.push_back(hole);

                    // Update counts
                    electronCount++;
                    holeCount++;
                }
            }
        }

        // ===== RECOMBINATION: Electrons and holes annihilate when close =====
        // Must happen in pairs to conserve charge

        float recombProb = CARRIER_LIFETIME_BASE;
        // Auger-like high-injection term
        recombProb += AUGER_COEFFICIENT * electronCount * holeCount;

        for (size_t i = 0; i < particles.size(); i++) {
            Particle& p = particles[i];
            if (!p.isFree || p.markedForDeletion || !p.isElectron) continue;

            float r = (float)rand() / RAND_MAX;
            if (r < recombProb * dt) {
                // Find nearby hole to recombine with
                int holeIdx = findRecombinationPartner((int)i, particles);

                if (holeIdx >= 0) {
                    // Recombination event!
                    // Electron returns to bound state at hole's location
                    // (simulating electron filling the hole)

                    p.isFree = false;
                    p.position = particles[holeIdx].homePosition;
                    p.homePosition = particles[holeIdx].homePosition;
                    p.velocity = { 0.0f, 0.0f };

                    // Delete the hole
                    particles[holeIdx].markedForDeletion = true;
                }
            }
        }

        // ===== CARRIER TRANSPORT =====

        for (Particle& p : particles) {
            if (p.markedForDeletion) continue;

            if (p.isFree) {
                // ----- DRIFT: Carriers move in electric field -----
                float mobility = calculateMobility(p.isElectron, T_k);
                float charge = p.isElectron ? -1.0f : +1.0f;

                // v_drift = mu * E * sign(charge)
                // Electrons move opposite to field, holes move with field
                float driftVelX = mobility * electricField * charge;

                // Velocity saturation at high fields
                float vsat = p.isElectron ? ELECTRON_VSAT_NORM : HOLE_VSAT_NORM;
                float E_crit = vsat / mobility;
                if (abs(electricField) > 0.01f) {
                    driftVelX = driftVelX / (1.0f + abs(electricField) / E_crit);
                }

                // ----- DIFFUSION: Carriers move from high to low concentration -----
                vec2 gradient = calculateDensityGradient(p.position, particles, p.isElectron);
                float diffCoeff = kT_sim * mobility * 0.5f;  // Einstein relation (scaled)

                vec2 diffusionVel = { 0.0f, 0.0f };
                if (length(gradient) > 0.1f) {
                    // Move against gradient (from high to low concentration)
                    diffusionVel = normalize(gradient) * (-diffCoeff);
                }

                // ----- THERMAL RANDOM WALK -----
                float thermalSpeed = sqrt(kT_sim) * 0.02f;
                float rx = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                float ry = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                vec2 thermalKick = { rx * thermalSpeed, ry * thermalSpeed };

                // Combine all velocity components
                vec2 targetVel = { driftVelX, 0.0f };
                targetVel += diffusionVel;
                targetVel += thermalKick;

                // Smooth velocity update (avoids jitter)
                p.velocity += (targetVel - p.velocity) * 0.2f;

                // Apply velocity
                float oldX = p.position.x;
                p.position += p.velocity * dt;

                // Track current: charge crossing center plane
                if ((oldX < 0.0f && p.position.x >= 0.0f) ||
                    (oldX >= 0.0f && p.position.x < 0.0f)) {
                    float chargeContribution = p.isElectron ? -1.0f : 1.0f;
                    if (p.position.x > oldX) chargeContribution *= 1.0f;
                    else chargeContribution *= -1.0f;
                    currentAccumulator += chargeContribution;
                }

                // ----- BOUNDARY CONDITIONS -----
                const float WALL_L = -0.9f, WALL_R = 0.9f;
                const float WALL_B = -0.8f, WALL_T = 0.8f;
                const float BOUNCE = 0.5f;

                if (p.position.x < WALL_L) {
                    p.position.x = WALL_L;
                    p.velocity.x = abs(p.velocity.x) * BOUNCE;
                }
                if (p.position.x > WALL_R) {
                    p.position.x = WALL_R;
                    p.velocity.x = -abs(p.velocity.x) * BOUNCE;
                }
                if (p.position.y < WALL_B) {
                    p.position.y = WALL_B;
                    p.velocity.y = abs(p.velocity.y) * BOUNCE;
                }
                if (p.position.y > WALL_T) {
                    p.position.y = WALL_T;
                    p.velocity.y = -abs(p.velocity.y) * BOUNCE;
                }
            }
            else if (p.isElectron) {
                // Bound electron: small thermal vibration around lattice site
                float vibrationAmp = 0.003f * (1.0f + temperature * 0.5f);
                float rx = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                float ry = ((float)rand() / RAND_MAX * 2.0f - 1.0f);

                vec2 vibration = { rx * vibrationAmp, ry * vibrationAmp };
                vec2 restore = (p.homePosition - p.position) * 0.3f;

                p.position += restore + vibration;
            }
        }

        // Remove deleted particles
        particles.erase(
            remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return p.markedForDeletion; }),
            particles.end()
        );

        // Update current measurement
        measureTimer += 1.0f;
        if (measureTimer >= MEASURE_INTERVAL) {
            currentMetric = currentAccumulator * (1000.0f / MEASURE_INTERVAL);  // Scale to uA
            currentAccumulator = 0.0f;
            measureTimer = 0.0f;
        }

        // ========== RENDERING ==========

        // Prepare GPU data
        vector<float> gpuData;
        gpuData.reserve(particles.size() * 5);

        for (const Particle& p : particles) {
            gpuData.push_back(p.position.x);
            gpuData.push_back(p.position.y);
            gpuData.push_back(0.0f);
            gpuData.push_back(p.isFree ? 1.0f : 0.0f);
            gpuData.push_back(p.isElectron ? 1.0f : 0.0f);
        }

        glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpuData.size() * sizeof(float), gpuData.data());

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glPointSize(5.0f);
        glDrawArrays(GL_POINTS, 0, (GLsizei)particles.size());

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}