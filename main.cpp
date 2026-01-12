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

// ---------- PHYSICAL CONSTANTS (REAL SILICON VALUES) ----------

const float Q_ELECTRON = 1.602e-19f;
const float BOLTZMANN_J = 1.381e-23f;
const float BOLTZMANN_EV = 8.617e-5f;
const float EPSILON_0 = 8.854e-14f;
const float EPSILON_SI = 11.7f;
const float EPSILON_SILICON = EPSILON_0 * EPSILON_SI;

const float BANDGAP_300K = 1.12f;
const float BANDGAP_COEFF_ALPHA = 4.73e-4f;
const float BANDGAP_COEFF_BETA = 636.0f;

const float NC_300K = 2.8e19f;
const float NV_300K = 1.04e19f;
const float NI_300K = 1.5e10f;

const float MU_ELECTRON_300K = 1350.0f;
const float MU_HOLE_300K = 480.0f;
const float MU_TEMP_EXP_E = 2.42f;
const float MU_TEMP_EXP_H = 2.20f;

const float VSAT_ELECTRON = 1.07e7f;
const float VSAT_HOLE = 8.37e6f;

const float TAU_N0 = 1e-6f;
const float TAU_P0 = 1e-6f;

// ---------- SIMULATION SCALING ----------

const float DEVICE_LENGTH_CM = 1.0e-4f;
const float SIM_LENGTH = 1.8f;
const float CM_TO_SIM = SIM_LENGTH / DEVICE_LENGTH_CM;
const float SIM_TO_CM = DEVICE_LENGTH_CM / SIM_LENGTH;

// ---------- EQUILIBRIUM CARRIER TARGETS ----------
// These define the target number of particles at equilibrium for different temperatures
// At 300K: target ~10-15 carriers (very few, intrinsic silicon)
// At 400K: target ~50-80 carriers
// At 500K: target ~150-200 carriers

const float EQUILIBRIUM_CARRIERS_300K = 12.0f;  // Target carrier count at 300K

// ---------- SIMULATION PARAMETERS ----------

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

float appliedVoltage = 0.0f;
float temperature = 300.0f;
float simulationSpeed = 1.0f;

float currentAccumulator = 0.0f;
float currentMetric = 0.0f;

const int IV_MAX_POINTS = 100;
float ivVoltages[IV_MAX_POINTS];
float ivCurrents[IV_MAX_POINTS];
int ivPointCount = 0;
bool ivSweepActive = false;
float ivSweepVoltage = -5.0f;
float ivSweepStep = 0.2f;
float ivSettleTime = 0.0f;
const float IV_SETTLE_DURATION = 0.5f;

int dopingMode = 0;
float dopingConcentration = 1.0e16f;
float displayDopingLog = 16.0f;

float Na_concentration = 1.0e16f;
float Nd_concentration = 1.0e16f;
float displayNaLog = 16.0f;
float displayNdLog = 16.0f;

const float PN_JUNCTION_POS = 0.0f;

struct DopingIon {
    vec2 position;
    bool isPositive;
};
vector<DopingIon> dopingIons;

// ---------- PARTICLE STRUCTURE ----------

struct Particle {
    vec2 position;
    vec2 velocity;
    vec2 homePosition;
    bool isFree;
    bool isElectron;
    bool markedForDeletion;
    bool isFromLattice;  // NEW: Track if this was originally a lattice electron
    int partnerId;

    Particle() : position({ 0,0 }), velocity({ 0,0 }), homePosition({ 0,0 }),
        isFree(false), isElectron(true), markedForDeletion(false),
        isFromLattice(false), partnerId(-1) {
    }
};

// Store the original lattice size so we can restore it
int originalLatticeSize = 0;

// ---------- PHYSICS FUNCTIONS ----------

float calculateBandgap(float T) {
    const float Eg_0 = 1.166f;
    return Eg_0 - (BANDGAP_COEFF_ALPHA * T * T) / (T + BANDGAP_COEFF_BETA);
}

float calculateIntrinsicConcentration(float T) {
    float Eg = calculateBandgap(T);
    float kT = BOLTZMANN_EV * T;
    if (kT < 0.001f) kT = 0.001f;

    float Nc = NC_300K * pow(T / 300.0f, 1.5f);
    float Nv = NV_300K * pow(T / 300.0f, 1.5f);

    float exponent = -Eg / (2.0f * kT);
    if (exponent < -50.0f) exponent = -50.0f;
    if (exponent > 50.0f) exponent = 50.0f;

    return sqrt(Nc) * sqrt(Nv) * exp(exponent);
}

// Calculate equilibrium carrier count for simulation based on temperature
// This scales with ni(T)/ni(300K) to give physically meaningful temperature dependence
float calculateEquilibriumCarrierCount(float T) {
    float ni_T = calculateIntrinsicConcentration(T);
    float ni_300 = calculateIntrinsicConcentration(300.0f);

    // Scale from our target at 300K
    // ni ratio can be huge (ni at 500K / ni at 300K ~ 10^4), so we use sqrt for gentler scaling
    float ratio = ni_T / ni_300;

    // Use power law scaling for visible but not explosive growth
    // At 300K: ratio = 1, count = 12
    // At 400K: ratio ~ 100, count ~ 12 * 100^0.3 ~ 48
    // At 500K: ratio ~ 10000, count ~ 12 * 10000^0.3 ~ 240
    float equilibrium = EQUILIBRIUM_CARRIERS_300K * pow(ratio, 0.3f);

    // Clamp to reasonable display range
    if (equilibrium < 5.0f) equilibrium = 5.0f;
    if (equilibrium > 350.0f) equilibrium = 350.0f;

    return equilibrium;
}

float calculateMobility(bool isElectron, float T) {
    float mu0 = isElectron ? MU_ELECTRON_300K : MU_HOLE_300K;
    float exp_val = isElectron ? MU_TEMP_EXP_E : MU_TEMP_EXP_H;
    return mu0 * pow(300.0f / T, exp_val);
}

float calculateMobilityWithDoping(bool isElectron, float T, float Ndoping) {
    float mu_lattice = calculateMobility(isElectron, T);
    float mu_min = isElectron ? 65.0f : 47.0f;
    float mu_max = mu_lattice;
    float N_ref = isElectron ? 8.5e16f : 6.3e16f;
    float alpha = isElectron ? 0.72f : 0.76f;

    return mu_min + (mu_max - mu_min) / (1.0f + pow(Ndoping / N_ref, alpha));
}

float calculateBuiltInPotential(float T, float Na, float Nd) {
    float ni = calculateIntrinsicConcentration(T);
    float kT_q = BOLTZMANN_EV * T;
    if (Na <= 0 || Nd <= 0 || ni <= 0) return 0.0f;
    return kT_q * log((Na * Nd) / (ni * ni));
}

float calculateDepletionWidth(float T, float Na, float Nd, float V_applied) {
    float V_bi = calculateBuiltInPotential(T, Na, Nd);
    float V_total = V_bi - V_applied;
    if (V_total < 0.01f) V_total = 0.01f;

    float numerator = 2.0f * EPSILON_SILICON * V_total * (Na + Nd);
    float denominator = Q_ELECTRON * Na * Nd;

    return sqrt(numerator / denominator);
}

void calculateDepletionWidths(float T, float Na, float Nd, float V_applied,
    float& xp, float& xn) {
    float W = calculateDepletionWidth(T, Na, Nd, V_applied);
    xp = W * Nd / (Na + Nd);
    xn = W * Na / (Na + Nd);
}

float calculateMaxDepletionFieldFromXn(float Nd, float xn) {
    return Q_ELECTRON * Nd * xn / EPSILON_SILICON;
}

float calculateMaxDepletionField(float T, float Na, float Nd, float V_applied) {
    float xp, xn;
    calculateDepletionWidths(T, Na, Nd, V_applied, xp, xn);
    return calculateMaxDepletionFieldFromXn(Nd, xn);
}

float calculateDiodeCurrent(float V, float T, float Na, float Nd, float Area) {
    float ni = calculateIntrinsicConcentration(T);
    float kT_q = BOLTZMANN_EV * T;

    float mu_n = calculateMobilityWithDoping(true, T, Nd);
    float mu_p = calculateMobilityWithDoping(false, T, Na);

    float Dn = kT_q * mu_n;
    float Dp = kT_q * mu_p;

    float Ln = sqrt(Dn * TAU_N0);
    float Lp = sqrt(Dp * TAU_P0);

    float Is = Area * Q_ELECTRON * ni * ni * (Dp / (Ln * Na) + Dn / (Lp * Nd));

    float n_ideality = 1.0f;
    float exponent = V / (n_ideality * kT_q);
    if (exponent > 40.0f) exponent = 40.0f;
    if (exponent < -40.0f) exponent = -40.0f;

    return Is * (exp(exponent) - 1.0f);
}

// Forward declaration
void setupDoping(int mode, vector<DopingIon>& ions, vector<Particle>& particles,
    float concentration, float T);

vec2 calculateLocalField(vec2 position, float appliedV, float T,
    float Na, float Nd, int mode) {
    vec2 field = { 0.0f, 0.0f };

    float E_applied = appliedV / DEVICE_LENGTH_CM;
    field.x = E_applied;

    if (mode == 3) {
        float x_cm = position.x * SIM_TO_CM;
        float V_bi = calculateBuiltInPotential(T, Na, Nd);
        float V_total = V_bi - appliedV;

        if (V_total > 0.01f) {
            float xp, xn;
            calculateDepletionWidths(T, Na, Nd, appliedV, xp, xn);

            if (x_cm > -xp && x_cm < xn) {
                float E_max = calculateMaxDepletionFieldFromXn(Nd, xn);

                if (x_cm < 0) {
                    field.x += -E_max * (1.0f + x_cm / xp);
                }
                else {
                    field.x += -E_max * (1.0f - x_cm / xn);
                }
            }
        }
    }

    return field;
}

// Reset lattice to original state
void resetLattice(vector<Particle>& particles, int latticeRows, int latticeCols) {
    particles.clear();
    particles.reserve(6000);

    for (int col = 0; col < latticeCols; col++) {
        for (int row = 0; row < latticeRows; row++) {
            Particle p;
            p.position.x = (float)col / (latticeCols - 1) * 1.7f - 0.85f;
            p.position.y = (float)row / (latticeRows - 1) * 1.5f - 0.75f;
            p.homePosition = p.position;
            p.velocity = { 0.0f, 0.0f };
            p.isFree = false;
            p.isElectron = true;
            p.markedForDeletion = false;
            p.isFromLattice = true;
            p.partnerId = -1;
            particles.push_back(p);
        }
    }
}

void setupDoping(int mode, vector<DopingIon>& ions, vector<Particle>& particles,
    float concentration, float T) {
    ions.clear();

    // Remove ONLY free carriers that are NOT from lattice (doping-added carriers)
    // Lattice electrons that became free should return to bound state
    for (Particle& p : particles) {
        if (p.isFree && p.isFromLattice) {
            p.isFree = false;
            // Position will be corrected by the restore force
        }
    }

    // Remove non-lattice free carriers (holes and doping-injected electrons)
    particles.erase(
        remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.isFree && !p.isFromLattice; }),
        particles.end()
    );

    if (mode == 0) return;

    int numVisualIons = (int)(log10(concentration) * 8);
    if (numVisualIons < 20) numVisualIons = 20;
    if (numVisualIons > 150) numVisualIons = 150;

    float ni = calculateIntrinsicConcentration(T);
    float majorityRatio = concentration / ni;
    int numFreeCarriers = (int)(log10(majorityRatio) * 15);
    if (numFreeCarriers < 10) numFreeCarriers = 10;
    if (numFreeCarriers > 200) numFreeCarriers = 200;

    if (mode == 1) {  // N-type
        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = true;
            ions.push_back(ion);
        }

        for (int i = 0; i < numFreeCarriers; i++) {
            Particle e;
            e.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            e.homePosition = e.position;
            e.velocity = { 0, 0 };
            e.isFree = true;
            e.isElectron = true;
            e.markedForDeletion = false;
            e.isFromLattice = false;  // This is a doping-added carrier
            particles.push_back(e);
        }
    }
    else if (mode == 2) {  // P-type
        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = false;
            ions.push_back(ion);
        }

        for (int i = 0; i < numFreeCarriers; i++) {
            Particle h;
            h.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            h.homePosition = h.position;
            h.velocity = { 0, 0 };
            h.isFree = true;
            h.isElectron = false;
            h.markedForDeletion = false;
            h.isFromLattice = false;
            particles.push_back(h);
        }
    }
    else if (mode == 3) {  // PN Junction
        float xp, xn;
        calculateDepletionWidths(T, concentration, concentration, 0.0f, xp, xn);
        float xp_sim = xp * CM_TO_SIM;
        float xn_sim = xn * CM_TO_SIM;

        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 0.85f - 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = false;
            ions.push_back(ion);
        }

        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = true;
            ions.push_back(ion);
        }

        for (int i = 0; i < numFreeCarriers; i++) {
            Particle h;
            h.position.x = -0.85f + ((float)rand() / RAND_MAX) * (0.85f - xp_sim - 0.1f);
            h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            h.homePosition = h.position;
            h.velocity = { 0, 0 };
            h.isFree = true;
            h.isElectron = false;
            h.markedForDeletion = false;
            h.isFromLattice = false;
            particles.push_back(h);
        }

        for (int i = 0; i < numFreeCarriers; i++) {
            Particle e;
            e.position.x = xn_sim + 0.1f + ((float)rand() / RAND_MAX) * (0.85f - xn_sim - 0.1f);
            e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            e.homePosition = e.position;
            e.velocity = { 0, 0 };
            e.isFree = true;
            e.isElectron = true;
            e.markedForDeletion = false;
            e.isFromLattice = false;
            particles.push_back(e);
        }
    }
}

void setupDopingPN(vector<DopingIon>& ions, vector<Particle>& particles,
    float Na, float Nd, float T) {
    ions.clear();

    // Same fix: return lattice electrons to bound state, remove others
    for (Particle& p : particles) {
        if (p.isFree && p.isFromLattice) {
            p.isFree = false;
        }
    }
    particles.erase(
        remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.isFree && !p.isFromLattice; }),
        particles.end()
    );

    float ni = calculateIntrinsicConcentration(T);

    float xp, xn;
    calculateDepletionWidths(T, Na, Nd, 0.0f, xp, xn);
    float xp_sim = xp * CM_TO_SIM;
    float xn_sim = xn * CM_TO_SIM;

    int numIonsP = (int)(log10(Na) * 8);
    int numIonsN = (int)(log10(Nd) * 8);
    if (numIonsP < 20) numIonsP = 20;
    if (numIonsN < 20) numIonsN = 20;
    if (numIonsP > 150) numIonsP = 150;
    if (numIonsN > 150) numIonsN = 150;

    float majorityRatioP = Na / ni;
    float majorityRatioN = Nd / ni;
    int numHoles = (int)(log10(majorityRatioP) * 15);
    int numElectrons = (int)(log10(majorityRatioN) * 15);
    if (numHoles < 10) numHoles = 10;
    if (numElectrons < 10) numElectrons = 10;
    if (numHoles > 200) numHoles = 200;
    if (numElectrons > 200) numElectrons = 200;

    for (int i = 0; i < numIonsP; i++) {
        DopingIon ion;
        ion.position.x = ((float)rand() / RAND_MAX) * 0.85f - 0.85f;
        ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        ion.isPositive = false;
        ions.push_back(ion);
    }

    for (int i = 0; i < numIonsN; i++) {
        DopingIon ion;
        ion.position.x = ((float)rand() / RAND_MAX) * 0.85f;
        ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        ion.isPositive = true;
        ions.push_back(ion);
    }

    for (int i = 0; i < numHoles; i++) {
        Particle h;
        h.position.x = -0.85f + ((float)rand() / RAND_MAX) * (0.85f - xp_sim - 0.1f);
        h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        h.homePosition = h.position;
        h.velocity = { 0, 0 };
        h.isFree = true;
        h.isElectron = false;
        h.markedForDeletion = false;
        h.isFromLattice = false;
        particles.push_back(h);
    }

    for (int i = 0; i < numElectrons; i++) {
        Particle e;
        e.position.x = xn_sim + 0.1f + ((float)rand() / RAND_MAX) * (0.85f - xn_sim - 0.1f);
        e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        e.homePosition = e.position;
        e.velocity = { 0, 0 };
        e.isFree = true;
        e.isElectron = true;
        e.markedForDeletion = false;
        e.isFromLattice = false;
        particles.push_back(e);
    }
}

struct DiffusionData {
    vec2 gradient;
    float localDensity;
};

DiffusionData calculateDiffusionData(vec2 position, const vector<Particle>& particles,
    bool forElectrons) {
    const float SAMPLE_RADIUS = 0.25f;

    float leftDensity = 0.0f, rightDensity = 0.0f;
    float upDensity = 0.0f, downDensity = 0.0f;
    float totalDensity = 0.0f;

    for (const Particle& p : particles) {
        if (!p.isFree || p.isElectron != forElectrons || p.markedForDeletion) continue;

        vec2 diff = p.position - position;
        float dist = length(diff);

        if (dist < SAMPLE_RADIUS && dist > 0.01f) {
            float weight = 1.0f - (dist / SAMPLE_RADIUS);
            totalDensity += weight;

            if (diff.x > 0) rightDensity += weight;
            else leftDensity += weight;

            if (diff.y > 0) upDensity += weight;
            else downDensity += weight;
        }
    }

    DiffusionData data;
    data.gradient = { rightDensity - leftDensity, upDensity - downDensity };
    data.localDensity = totalDensity + 0.1f;

    return data;
}

int findRecombinationPartner(int myIndex, const vector<Particle>& particles, float radius) {
    const Particle& me = particles[myIndex];
    if (!me.isFree || me.markedForDeletion) return -1;

    float minDist = radius;
    int bestPartner = -1;

    for (size_t i = 0; i < particles.size(); i++) {
        if (i == (size_t)myIndex) continue;
        const Particle& other = particles[i];

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
            FragColor = vec4(0.2, 0.8, 1.0, 1.0);
        else
            FragColor = vec4(1.0, 0.3, 0.3, 1.0);
    } else {
        FragColor = vec4(0.25, 0.3, 0.4, 1.0);
    }
}
)";

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// ---------- MAIN ----------

int main()
{
    srand(static_cast<unsigned int>(time(0)));

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
        "Semiconductor Physics Simulation - Fixed Equilibrium", NULL, NULL);
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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

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

    // Create crystal lattice
    vector<Particle> particles;
    const int LATTICE_ROWS = 25;
    const int LATTICE_COLS = 40;
    const int LATTICE_SIZE = LATTICE_ROWS * LATTICE_COLS;
    originalLatticeSize = LATTICE_SIZE;

    particles.reserve(6000);

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
            p.isFromLattice = true;
            p.partnerId = -1;
            particles.push_back(p);
        }
    }

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, 6000 * 5 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glEnable(GL_PROGRAM_POINT_SIZE);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ===== CONTROL PANEL =====
        ImGui::Begin("Semiconductor Lab Controls");

        ImGui::Text("Device Parameters");
        ImGui::SliderFloat("Applied Voltage (V)", &appliedVoltage, -5.0f, 5.0f);
        ImGui::SliderFloat("Temperature (K)", &temperature, 200.0f, 500.0f);
        ImGui::SliderFloat("Simulation Speed", &simulationSpeed, 0.1f, 3.0f);

        if (ImGui::Button("Reset All")) {
            // Full reset: restore lattice to original state
            resetLattice(particles, LATTICE_ROWS, LATTICE_COLS);
            dopingIons.clear();
            dopingMode = 0;
            currentAccumulator = 0.0f;
            currentMetric = 0.0f;
            ivPointCount = 0;
        }

        // Show equilibrium target
        float eqTarget = calculateEquilibriumCarrierCount(temperature);
        ImGui::Text("Equilibrium target: ~%.0f carriers", eqTarget);

        ImGui::Separator();
        ImGui::Text("Doping Configuration");

        const char* dopingModes[] = { "Intrinsic", "N-Type", "P-Type", "P-N Junction" };
        int prevDopingMode = dopingMode;
        ImGui::Combo("Device Type", &dopingMode, dopingModes, 4);

        if (dopingMode > 0 && dopingMode < 3) {
            float prevLog = displayDopingLog;
            ImGui::SliderFloat("Doping (10^x cm-3)", &displayDopingLog, 14.0f, 18.0f);
            dopingConcentration = pow(10.0f, displayDopingLog);

            ImGui::Text("N_doping = %.2e cm^-3", dopingConcentration);

            if (dopingMode != prevDopingMode || displayDopingLog != prevLog) {
                setupDoping(dopingMode, dopingIons, particles, dopingConcentration, temperature);
                ivPointCount = 0;
            }
        }
        else if (dopingMode == 3) {
            float prevNaLog = displayNaLog;
            float prevNdLog = displayNdLog;

            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "P-side (Acceptors):");
            ImGui::SliderFloat("Na (10^x cm-3)", &displayNaLog, 14.0f, 18.0f);
            Na_concentration = pow(10.0f, displayNaLog);
            ImGui::Text("Na = %.2e cm^-3", Na_concentration);

            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "N-side (Donors):");
            ImGui::SliderFloat("Nd (10^x cm-3)", &displayNdLog, 14.0f, 18.0f);
            Nd_concentration = pow(10.0f, displayNdLog);
            ImGui::Text("Nd = %.2e cm^-3", Nd_concentration);

            if (dopingMode != prevDopingMode || displayNaLog != prevNaLog || displayNdLog != prevNdLog) {
                setupDopingPN(dopingIons, particles, Na_concentration, Nd_concentration, temperature);
                ivPointCount = 0;
            }
        }
        else if (dopingMode != prevDopingMode) {
            setupDoping(dopingMode, dopingIons, particles, dopingConcentration, temperature);
            ivPointCount = 0;
        }

        if (dopingMode == 3) {
            float V_bi = calculateBuiltInPotential(temperature, Na_concentration, Nd_concentration);
            float xp, xn;
            calculateDepletionWidths(temperature, Na_concentration, Nd_concentration,
                appliedVoltage, xp, xn);
            float W_total = xp + xn;
            float E_max = calculateMaxDepletionField(temperature, Na_concentration,
                Nd_concentration, appliedVoltage);

            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "P-N Junction Properties:");
            ImGui::Text("Built-in Potential: %.3f V", V_bi);
            ImGui::Text("Depletion Width: %.2e cm", W_total);
            ImGui::Text("Max Field: %.2e V/cm", E_max);
        }

        ImGui::Separator();
        ImGui::Text("Physical Properties");

        float kT_display = BOLTZMANN_EV * temperature;
        float ni_display = calculateIntrinsicConcentration(temperature);
        float Eg = calculateBandgap(temperature);
        float mu_e = calculateMobility(true, temperature);
        float mu_h = calculateMobility(false, temperature);

        ImGui::Text("Temperature: %.0f K", temperature);
        ImGui::Text("kT: %.4f eV", kT_display);
        ImGui::Text("Bandgap: %.3f eV", Eg);
        ImGui::Text("n_i: %.2e cm^-3", ni_display);
        ImGui::Text("Mobility e/h: %.0f / %.0f cm2/(V.s)", mu_e, mu_h);

        ImGui::Separator();
        ImGui::Text("Carrier Statistics");

        int freeElectrons = 0, holes = 0, boundElectrons = 0;
        int latticeElectrons = 0;
        for (const Particle& p : particles) {
            if (p.markedForDeletion) continue;
            if (p.isFree) {
                if (p.isElectron) freeElectrons++;
                else holes++;
            }
            else {
                boundElectrons++;
                if (p.isFromLattice) latticeElectrons++;
            }
        }

        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Free Electrons: %d", freeElectrons);
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Holes: %d", holes);
        ImGui::Text("Bound (lattice): %d / %d", latticeElectrons, LATTICE_SIZE);
        ImGui::Text("Total Free: %d (target: ~%.0f)", freeElectrons + holes, eqTarget * 2);

        ImGui::Separator();
        ImGui::Text("Current: %.2f (arb. units)", currentMetric);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::End();

        // ===== I-V CURVE WINDOW =====
        ImGui::Begin("I-V Characteristic");

        if (!ivSweepActive) {
            if (ImGui::Button("Start Sweep")) {
                ivSweepActive = true;
                ivSweepVoltage = -5.0f;
                ivPointCount = 0;
                ivSettleTime = 0.0f;
                appliedVoltage = ivSweepVoltage;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear")) ivPointCount = 0;
            ImGui::SameLine();
            if (ImGui::Button("Add Point")) {
                if (ivPointCount < IV_MAX_POINTS) {
                    ivVoltages[ivPointCount] = appliedVoltage;
                    ivCurrents[ivPointCount] = currentMetric;
                    ivPointCount++;
                }
            }
        }
        else {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Sweeping: %.1f V", ivSweepVoltage);
            ImGui::SameLine();
            if (ImGui::Button("Stop")) ivSweepActive = false;
        }

        ImGui::SliderFloat("Step", &ivSweepStep, 0.1f, 1.0f);

        if (ivPointCount > 0) {
            float minI = ivCurrents[0], maxI = ivCurrents[0];
            float minV = ivVoltages[0], maxV = ivVoltages[0];
            for (int i = 0; i < ivPointCount; i++) {
                if (ivCurrents[i] < minI) minI = ivCurrents[i];
                if (ivCurrents[i] > maxI) maxI = ivCurrents[i];
                if (ivVoltages[i] < minV) minV = ivVoltages[i];
                if (ivVoltages[i] > maxV) maxV = ivVoltages[i];
            }

            float iRange = maxI - minI;
            if (iRange < 1.0f) iRange = 1.0f;
            minI -= iRange * 0.1f;
            maxI += iRange * 0.1f;

            ImVec2 plotSize(350, 200);
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            drawList->AddRectFilled(plotPos,
                ImVec2(plotPos.x + plotSize.x, plotPos.y + plotSize.y),
                IM_COL32(30, 30, 40, 255));

            float y_zero = plotPos.y + plotSize.y - (0 - minI) / (maxI - minI + 0.001f) * plotSize.y;
            if (y_zero > plotPos.y && y_zero < plotPos.y + plotSize.y) {
                drawList->AddLine(ImVec2(plotPos.x, y_zero),
                    ImVec2(plotPos.x + plotSize.x, y_zero), IM_COL32(80, 80, 80, 255));
            }

            float v0_x = plotPos.x + (0 - minV) / (maxV - minV + 0.001f) * plotSize.x;
            if (v0_x > plotPos.x && v0_x < plotPos.x + plotSize.x) {
                drawList->AddLine(ImVec2(v0_x, plotPos.y),
                    ImVec2(v0_x, plotPos.y + plotSize.y), IM_COL32(80, 80, 80, 255));
            }

            for (int i = 0; i < ivPointCount; i++) {
                float x = plotPos.x + (ivVoltages[i] - minV) / (maxV - minV + 0.001f) * plotSize.x;
                float y = plotPos.y + plotSize.y - (ivCurrents[i] - minI) / (maxI - minI + 0.001f) * plotSize.y;
                x = fmax(plotPos.x, fmin(x, plotPos.x + plotSize.x));
                y = fmax(plotPos.y, fmin(y, plotPos.y + plotSize.y));

                drawList->AddCircleFilled(ImVec2(x, y), 4.0f, IM_COL32(100, 255, 100, 255));

                if (i > 0) {
                    float px = plotPos.x + (ivVoltages[i - 1] - minV) / (maxV - minV + 0.001f) * plotSize.x;
                    float py = plotPos.y + plotSize.y - (ivCurrents[i - 1] - minI) / (maxI - minI + 0.001f) * plotSize.y;
                    px = fmax(plotPos.x, fmin(px, plotPos.x + plotSize.x));
                    py = fmax(plotPos.y, fmin(py, plotPos.y + plotSize.y));
                    drawList->AddLine(ImVec2(px, py), ImVec2(x, y), IM_COL32(100, 255, 100, 180), 2.0f);
                }
            }

            drawList->AddRect(plotPos,
                ImVec2(plotPos.x + plotSize.x, plotPos.y + plotSize.y),
                IM_COL32(100, 100, 100, 255));

            ImGui::Dummy(plotSize);
            ImGui::Text("V: %.1f to %.1f | I: %.1f to %.1f", minV, maxV, minI, maxI);
        }

        ImGui::End();

        // ========== PHYSICS SIMULATION ==========

        float actualDt = ImGui::GetIO().DeltaTime;
        if (actualDt > 0.05f) actualDt = 0.05f;
        if (actualDt < 0.001f) actualDt = 0.001f;
        float dt = actualDt * simulationSpeed;

        // I-V sweep logic
        if (ivSweepActive) {
            ivSettleTime += actualDt;
            appliedVoltage = ivSweepVoltage;

            if (ivSettleTime >= IV_SETTLE_DURATION) {
                if (ivPointCount < IV_MAX_POINTS) {
                    ivVoltages[ivPointCount] = ivSweepVoltage;
                    ivCurrents[ivPointCount] = currentMetric;
                    ivPointCount++;
                }
                ivSweepVoltage += ivSweepStep;
                ivSettleTime = 0.0f;

                if (ivSweepVoltage > 5.0f) {
                    ivSweepActive = false;
                    appliedVoltage = 0.0f;
                }
            }
        }

        float kT_eV = BOLTZMANN_EV * temperature;

        // Count current carriers
        int electronCount = 0, holeCount = 0;
        for (const Particle& p : particles) {
            if (p.isFree && !p.markedForDeletion) {
                if (p.isElectron) electronCount++;
                else holeCount++;
            }
        }

        // ===== GENERATION/RECOMBINATION WITH PROPER EQUILIBRIUM =====
        //
        // Key insight: At equilibrium, generation rate = recombination rate
        // We use a TARGET-BASED approach:
        //   - Calculate equilibrium carrier count based on temperature
        //   - If current count < target: favor generation
        //   - If current count > target: favor recombination
        //
        // This ensures the system naturally reaches and maintains equilibrium

        float equilibriumTarget = calculateEquilibriumCarrierCount(temperature);
        float currentCarrierCount = (float)(electronCount + holeCount) / 2.0f;  // Average of e and h

        // Deviation from equilibrium: positive = too many carriers, negative = too few
        float deviation = currentCarrierCount - equilibriumTarget;
        float deviationRatio = deviation / (equilibriumTarget + 1.0f);

        // Base rates (these get modified by deviation)
        float baseGenerationProb = 0.0005f;
        float baseRecombinationProb = 0.0005f;

        // Adjust rates based on deviation from equilibrium
        // When too few carriers: boost generation, reduce recombination
        // When too many carriers: reduce generation, boost recombination
        float generationProb = baseGenerationProb * (1.0f - deviationRatio * 2.0f);
        float recombinationProb = baseRecombinationProb * (1.0f + deviationRatio * 2.0f);

        // Clamp to reasonable ranges
        generationProb = fmax(0.0f, fmin(generationProb, 0.002f));
        recombinationProb = fmax(0.0f, fmin(recombinationProb, 0.01f));

        // Also limit total carriers to prevent runaway
        bool canGenerate = (particles.size() < 5500) && (electronCount + holeCount < 400);

        // GENERATION: Create electron-hole pairs from lattice
        for (size_t i = 0; i < particles.size() && i < (size_t)LATTICE_SIZE; i++) {
            Particle& p = particles[i];

            if (!p.isFree && p.isElectron && p.isFromLattice && canGenerate) {
                if ((float)rand() / RAND_MAX < generationProb * dt) {
                    // Excite this lattice electron
                    p.isFree = true;
                    p.velocity = { 0.0f, 0.0f };

                    // Create corresponding hole
                    Particle hole;
                    hole.position = p.position;
                    hole.homePosition = p.homePosition;
                    hole.velocity = { 0.0f, 0.0f };
                    hole.isFree = true;
                    hole.isElectron = false;
                    hole.markedForDeletion = false;
                    hole.isFromLattice = false;  // Holes are never "from lattice"
                    particles.push_back(hole);
                }
            }
        }

        // RECOMBINATION: Electron-hole pairs annihilate
        float recombRadius = 0.15f;

        for (size_t i = 0; i < particles.size(); i++) {
            Particle& p = particles[i];
            if (!p.isFree || p.markedForDeletion || !p.isElectron) continue;

            if ((float)rand() / RAND_MAX < recombinationProb * dt) {
                int holeIdx = findRecombinationPartner((int)i, particles, recombRadius);

                if (holeIdx >= 0) {
                    // Electron returns to bound state
                    p.isFree = false;
                    p.position = particles[holeIdx].homePosition;
                    p.homePosition = particles[holeIdx].homePosition;
                    p.velocity = { 0.0f, 0.0f };

                    // Hole is deleted
                    particles[holeIdx].markedForDeletion = true;
                }
            }
        }

        // ===== P-N JUNCTION SPECIFIC PHYSICS =====
        if (dopingMode == 3 && appliedVoltage > 0.1f) {
            // Forward bias injection
            float kT_q = BOLTZMANN_EV * temperature;
            float exponent = appliedVoltage / (kT_q * 2.0f);
            if (exponent > 10.0f) exponent = 10.0f;

            float injectionProb = 0.001f * (exp(exponent) - 1.0f);
            injectionProb = fmin(injectionProb, 0.3f);

            for (Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion) continue;

                if (p.isElectron && p.position.x > 0.05f && p.position.x < 0.5f) {
                    if ((float)rand() / RAND_MAX < injectionProb * dt) {
                        p.position.x = -0.05f - ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = -0.02f;
                    }
                }

                if (!p.isElectron && p.position.x < -0.05f && p.position.x > -0.5f) {
                    if ((float)rand() / RAND_MAX < injectionProb * dt) {
                        p.position.x = 0.05f + ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = 0.02f;
                    }
                }
            }
        }

        if (dopingMode == 3 && appliedVoltage < -0.1f) {
            // Reverse bias sweep-out
            float sweepProb = 0.01f * fabs(appliedVoltage);
            sweepProb = fmin(sweepProb, 0.1f);

            for (Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion) continue;

                if (p.isElectron && p.position.x > -0.3f && p.position.x < 0.0f) {
                    if ((float)rand() / RAND_MAX < sweepProb * dt) {
                        p.position.x = 0.5f + ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = 0.02f;
                    }
                }

                if (!p.isElectron && p.position.x > 0.0f && p.position.x < 0.3f) {
                    if ((float)rand() / RAND_MAX < sweepProb * dt) {
                        p.position.x = -0.5f - ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = -0.02f;
                    }
                }
            }
        }

        // ===== CARRIER TRANSPORT =====
        float doping_for_mobility = dopingMode > 0 ? dopingConcentration : 1e10f;
        float mu_electron = calculateMobilityWithDoping(true, temperature, doping_for_mobility);
        float mu_hole = calculateMobilityWithDoping(false, temperature, doping_for_mobility);

        float mu_electron_sim = mu_electron * 1e-6f;
        float mu_hole_sim = mu_hole * 1e-6f;

        float vsat_electron_sim = VSAT_ELECTRON * 1e-7f;
        float vsat_hole_sim = VSAT_HOLE * 1e-7f;

        for (Particle& p : particles) {
            if (p.markedForDeletion) continue;

            if (p.isFree) {
                float mu_sim = p.isElectron ? mu_electron_sim : mu_hole_sim;
                float vsat_sim = p.isElectron ? vsat_electron_sim : vsat_hole_sim;

                float Na_for_field = (dopingMode == 3) ? Na_concentration : dopingConcentration;
                float Nd_for_field = (dopingMode == 3) ? Nd_concentration : dopingConcentration;
                vec2 E_field = calculateLocalField(p.position, appliedVoltage, temperature,
                    Na_for_field, Nd_for_field, dopingMode);

                float E_scale = 1e-4f;
                float E_sim_x = E_field.x * E_scale;
                float E_sim_y = E_field.y * E_scale;
                float E_magnitude = sqrt(E_sim_x * E_sim_x + E_sim_y * E_sim_y);

                float charge = p.isElectron ? -1.0f : 1.0f;

                float driftVelX = 0.0f, driftVelY = 0.0f;
                if (E_magnitude > 1e-10f) {
                    float muE = mu_sim * E_magnitude;
                    float v_magnitude = muE / (1.0f + muE / vsat_sim);

                    float E_norm_x = E_sim_x / E_magnitude;
                    float E_norm_y = E_sim_y / E_magnitude;
                    driftVelX = v_magnitude * E_norm_x * charge;
                    driftVelY = v_magnitude * E_norm_y * charge;
                }

                DiffusionData diffData = calculateDiffusionData(p.position, particles, p.isElectron);
                float D_sim = kT_eV * mu_sim * 0.5f;

                vec2 diffusionVel = { 0.0f, 0.0f };
                float gradientMag = length(diffData.gradient);
                if (gradientMag > 0.05f && diffData.localDensity > 0.1f) {
                    diffusionVel.x = -D_sim * (diffData.gradient.x / diffData.localDensity);
                    diffusionVel.y = -D_sim * (diffData.gradient.y / diffData.localDensity);

                    float maxDiffVel = 0.02f;
                    float diffVelMag = length(diffusionVel);
                    if (diffVelMag > maxDiffVel) {
                        diffusionVel.x *= maxDiffVel / diffVelMag;
                        diffusionVel.y *= maxDiffVel / diffVelMag;
                    }
                }

                float thermalSpeed = sqrt(kT_eV) * 0.05f;
                float rx = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                float ry = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                vec2 thermalKick = { rx * thermalSpeed, ry * thermalSpeed };

                vec2 targetVel = { driftVelX + diffusionVel.x + thermalKick.x,
                                   driftVelY + diffusionVel.y + thermalKick.y };

                p.velocity += (targetVel - p.velocity) * 0.15f;
                p.position += p.velocity * dt;

                if (dopingMode != 3) {
                    currentAccumulator += charge * driftVelX * 1000.0f;
                }

                const float WALL_L = -0.9f, WALL_R = 0.9f;
                const float WALL_B = -0.8f, WALL_T = 0.8f;

                if (p.position.x < WALL_L) { p.position.x = WALL_L; p.velocity.x *= -0.5f; }
                if (p.position.x > WALL_R) { p.position.x = WALL_R; p.velocity.x *= -0.5f; }
                if (p.position.y < WALL_B) { p.position.y = WALL_B; p.velocity.y *= -0.5f; }
                if (p.position.y > WALL_T) { p.position.y = WALL_T; p.velocity.y *= -0.5f; }

            }
            else if (p.isElectron) {
                float vibrationAmp = 0.002f * sqrt(temperature / 300.0f);
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

        // Diode current for P-N junction
        if (dopingMode == 3) {
            float kT_q = BOLTZMANN_EV * temperature;
            float n_ideality = 1.8f;
            float Is_normalized = 0.001f;

            float Vt = n_ideality * kT_q;
            float diodeCurrent;

            if (appliedVoltage > 0.0f) {
                float exponent = appliedVoltage / Vt;
                if (exponent > 10.0f) exponent = 10.0f;
                diodeCurrent = Is_normalized * (exp(exponent) - 1.0f);
            }
            else {
                diodeCurrent = -Is_normalized;
            }

            currentAccumulator = diodeCurrent * 500.0f;
        }

        currentMetric = currentMetric * 0.95f + currentAccumulator * 0.05f;
        currentAccumulator = 0.0f;

        // ========== RENDERING ==========

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

        // Render doping ions
        if (!dopingIons.empty()) {
            ImDrawList* drawList = ImGui::GetBackgroundDrawList();
            int windowWidth, windowHeight;
            glfwGetWindowSize(window, &windowWidth, &windowHeight);

            for (const DopingIon& ion : dopingIons) {
                float screenX = (ion.position.x + 1.0f) * 0.5f * windowWidth;
                float screenY = (1.0f - (ion.position.y + 1.0f) * 0.5f) * windowHeight;

                float size = 6.0f;
                ImU32 color = ion.isPositive ?
                    IM_COL32(100, 150, 255, 200) : IM_COL32(255, 150, 100, 200);

                drawList->AddRectFilled(
                    ImVec2(screenX - size, screenY - size),
                    ImVec2(screenX + size, screenY + size), color);

                ImU32 symbolColor = IM_COL32(255, 255, 255, 255);
                if (ion.isPositive) {
                    drawList->AddLine(ImVec2(screenX - 3, screenY),
                        ImVec2(screenX + 3, screenY), symbolColor, 2.0f);
                    drawList->AddLine(ImVec2(screenX, screenY - 3),
                        ImVec2(screenX, screenY + 3), symbolColor, 2.0f);
                }
                else {
                    drawList->AddLine(ImVec2(screenX - 3, screenY),
                        ImVec2(screenX + 3, screenY), symbolColor, 2.0f);
                }
            }

            if (dopingMode == 3) {
                float xp, xn;
                calculateDepletionWidths(temperature, Na_concentration,
                    Nd_concentration, appliedVoltage, xp, xn);

                float xp_sim = xp * CM_TO_SIM;
                float xn_sim = xn * CM_TO_SIM;

                float screenXp = (-xp_sim + 1.0f) * 0.5f * windowWidth;
                float screenXn = (xn_sim + 1.0f) * 0.5f * windowWidth;
                float screenXjunction = (0.0f + 1.0f) * 0.5f * windowWidth;

                drawList->AddLine(ImVec2(screenXp, 50), ImVec2(screenXp, windowHeight - 50),
                    IM_COL32(255, 255, 0, 100), 2.0f);
                drawList->AddLine(ImVec2(screenXn, 50), ImVec2(screenXn, windowHeight - 50),
                    IM_COL32(255, 255, 0, 100), 2.0f);
                drawList->AddLine(ImVec2(screenXjunction, 50), ImVec2(screenXjunction, windowHeight - 50),
                    IM_COL32(255, 255, 255, 50), 1.0f);
            }
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

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