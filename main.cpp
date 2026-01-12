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

// Fundamental constants
const float Q_ELECTRON = 1.602e-19f;      // Coulombs
const float BOLTZMANN_J = 1.381e-23f;     // J/K
const float BOLTZMANN_EV = 8.617e-5f;     // eV/K
const float EPSILON_0 = 8.854e-14f;       // F/cm
const float EPSILON_SI = 11.7f;           // Silicon relative permittivity
const float EPSILON_SILICON = EPSILON_0 * EPSILON_SI;  // F/cm

// Silicon bandgap and intrinsic properties
const float BANDGAP_300K = 1.12f;         // eV at 300K
const float BANDGAP_COEFF_ALPHA = 4.73e-4f;  // eV/K (Varshni parameter)
const float BANDGAP_COEFF_BETA = 636.0f;     // K (Varshni parameter)

// Effective density of states at 300K
const float NC_300K = 2.8e19f;            // cm^-3 (conduction band)
const float NV_300K = 1.04e19f;           // cm^-3 (valence band)

// Intrinsic carrier concentration at 300K
const float NI_300K = 1.5e10f;            // cm^-3

// Mobility at 300K (low-field, undoped)
const float MU_ELECTRON_300K = 1350.0f;   // cm²/(V·s)
const float MU_HOLE_300K = 480.0f;        // cm²/(V·s)

// Mobility temperature exponents (lattice scattering)
const float MU_TEMP_EXP_E = 2.42f;
const float MU_TEMP_EXP_H = 2.20f;

// Saturation velocities
const float VSAT_ELECTRON = 1.07e7f;      // cm/s
const float VSAT_HOLE = 8.37e6f;          // cm/s

// Critical fields for velocity saturation (Si at 300K)
const float ECRIT_ELECTRON = 8.0e3f;      // V/cm
const float ECRIT_HOLE = 1.95e4f;         // V/cm

// SRH recombination lifetimes (typical for moderately doped Si)
const float TAU_N0 = 1e-6f;               // seconds (electron lifetime)
const float TAU_P0 = 1e-6f;               // seconds (hole lifetime)

// Auger coefficients
const float AUGER_CN = 2.8e-31f;          // cm^6/s
const float AUGER_CP = 9.9e-32f;          // cm^6/s

// ---------- SIMULATION SCALING ----------
// We need to map real physics to visual simulation
// Simulation uses normalized coordinates [-1, 1] representing a device

const float DEVICE_LENGTH_CM = 1.0e-4f;   // 1 micrometer device
const float SIM_LENGTH = 1.8f;            // Simulation coordinate range
const float CM_TO_SIM = SIM_LENGTH / DEVICE_LENGTH_CM;
const float SIM_TO_CM = DEVICE_LENGTH_CM / SIM_LENGTH;

// Time scaling: 1 simulation second = REAL_TIME_SCALE real seconds
const float REAL_TIME_SCALE = 1.0e-12f;   // 1 sim second = 1 picosecond

// Carrier scaling: each particle represents CARRIERS_PER_PARTICLE real carriers
// For visualization, we work with much fewer particles than real carrier counts
const float CARRIERS_PER_PARTICLE = 1.0e8f;

// ---------- SIMULATION PARAMETERS ----------

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// Simulation state
float appliedVoltage = 0.0f;
float temperature = 300.0f;               // Now in actual Kelvin
float simulationSpeed = 1.0f;

// Current measurement
float currentAccumulator = 0.0f;
float currentMetric = 0.0f;

// I-V Curve data
const int IV_MAX_POINTS = 100;
float ivVoltages[IV_MAX_POINTS];
float ivCurrents[IV_MAX_POINTS];
int ivPointCount = 0;
bool ivSweepActive = false;
float ivSweepVoltage = -5.0f;
float ivSweepStep = 0.2f;
float ivSettleTime = 0.0f;
const float IV_SETTLE_DURATION = 0.5f;

// Doping configuration
int dopingMode = 0;                       // 0=intrinsic, 1=n-type, 2=p-type, 3=pn junction
float dopingConcentration = 1.0e16f;      // cm^-3 (for n-type and p-type modes)
float displayDopingLog = 16.0f;           // For slider: log10 of doping

// Separate Na and Nd for asymmetric P-N junction
float Na_concentration = 1.0e16f;         // Acceptor concentration (P-side)
float Nd_concentration = 1.0e16f;         // Donor concentration (N-side)
float displayNaLog = 16.0f;               // For slider: log10 of Na
float displayNdLog = 16.0f;               // For slider: log10 of Nd

const float PN_JUNCTION_POS = 0.0f;

// Fixed ions for doping visualization
struct DopingIon {
    vec2 position;
    bool isPositive;  // true = donor (n-type), false = acceptor (p-type)
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
    int partnerId;

    Particle() : position({ 0,0 }), velocity({ 0,0 }), homePosition({ 0,0 }),
        isFree(false), isElectron(true), markedForDeletion(false), partnerId(-1) {
    }
};

// ---------- PHYSICS FUNCTIONS ----------

// Temperature-dependent bandgap (Varshni equation)
// Eg(T) = Eg(0) - α*T²/(T+β)
// For Si: Eg(0) = 1.166 eV, α = 4.73×10⁻⁴ eV/K, β = 636 K
float calculateBandgap(float T) {
    const float Eg_0 = 1.166f;  // Bandgap at 0K
    return Eg_0 - (BANDGAP_COEFF_ALPHA * T * T) / (T + BANDGAP_COEFF_BETA);
}

// Temperature-dependent intrinsic carrier concentration
// n_i = sqrt(Nc * Nv) * exp(-Eg / 2kT)
float calculateIntrinsicConcentration(float T) {
    float Eg = calculateBandgap(T);
    float kT = BOLTZMANN_EV * T;

    // Prevent division by zero or very small kT
    if (kT < 0.001f) kT = 0.001f;

    // Nc and Nv scale as T^(3/2)
    float Nc = NC_300K * pow(T / 300.0f, 1.5f);
    float Nv = NV_300K * pow(T / 300.0f, 1.5f);

    // Calculate sqrt(Nc) * sqrt(Nv) separately to avoid overflow
    float sqrt_Nc = sqrt(Nc);
    float sqrt_Nv = sqrt(Nv);

    // Clamp the exponent to prevent overflow/underflow
    float exponent = -Eg / (2.0f * kT);
    if (exponent < -50.0f) exponent = -50.0f;  // exp(-50) ≈ 0
    if (exponent > 50.0f) exponent = 50.0f;    // exp(50) is huge but finite

    return sqrt_Nc * sqrt_Nv * exp(exponent);
}

// Temperature-dependent mobility (lattice scattering)
float calculateMobility(bool isElectron, float T) {
    float mu0 = isElectron ? MU_ELECTRON_300K : MU_HOLE_300K;
    float exp_val = isElectron ? MU_TEMP_EXP_E : MU_TEMP_EXP_H;
    return mu0 * pow(300.0f / T, exp_val);
}

// Doping-dependent mobility (empirical Caughey-Thomas for doping)
float calculateMobilityWithDoping(bool isElectron, float T, float Ndoping) {
    float mu_lattice = calculateMobility(isElectron, T);

    // Empirical mobility reduction due to impurity scattering
    // μ = μ_min + (μ_max - μ_min) / (1 + (N/N_ref)^α)
    float mu_min = isElectron ? 65.0f : 47.0f;
    float mu_max = mu_lattice;
    float N_ref = isElectron ? 8.5e16f : 6.3e16f;
    float alpha = isElectron ? 0.72f : 0.76f;

    return mu_min + (mu_max - mu_min) / (1.0f + pow(Ndoping / N_ref, alpha));
}

// Field-dependent velocity (Caughey-Thomas high-field model)
// v = μE / (1 + μE/vsat)
// This gives v → vsat as E → ∞ (smooth saturation)
float calculateFieldDependentVelocity(float E, bool isElectron, float mu, float vsat) {
    // Caughey-Thomas model: v = μE / (1 + (μE/vsat))
    float muE = mu * fabs(E);
    return muE / (1.0f + muE / vsat);
}

// Built-in potential for PN junction
// V_bi = (kT/q) * ln(Na * Nd / ni²)
float calculateBuiltInPotential(float T, float Na, float Nd) {
    float ni = calculateIntrinsicConcentration(T);
    float kT_q = BOLTZMANN_EV * T;  // kT/q in volts (since we use eV)

    if (Na <= 0 || Nd <= 0 || ni <= 0) return 0.0f;

    return kT_q * log((Na * Nd) / (ni * ni));
}

// Depletion width calculation
// W = sqrt(2 * ε * V_bi * (Na + Nd) / (q * Na * Nd))
float calculateDepletionWidth(float T, float Na, float Nd, float V_applied) {
    float V_bi = calculateBuiltInPotential(T, Na, Nd);
    float V_total = V_bi - V_applied;  // Reverse bias increases, forward decreases

    if (V_total < 0.01f) V_total = 0.01f;  // Prevent negative/zero

    float numerator = 2.0f * EPSILON_SILICON * V_total * (Na + Nd);
    float denominator = Q_ELECTRON * Na * Nd;

    return sqrt(numerator / denominator);
}

// Depletion widths on each side
void calculateDepletionWidths(float T, float Na, float Nd, float V_applied,
    float& xp, float& xn) {
    float W = calculateDepletionWidth(T, Na, Nd, V_applied);
    // xp = W * Nd / (Na + Nd), xn = W * Na / (Na + Nd)
    xp = W * Nd / (Na + Nd);
    xn = W * Na / (Na + Nd);
}

// Electric field in depletion region (triangular approximation)
// Maximum field at junction: E_max = q * Nd * xn / ε
// Can be called with pre-calculated xn to avoid redundant depletion width calculation
float calculateMaxDepletionFieldFromXn(float Nd, float xn) {
    return Q_ELECTRON * Nd * xn / EPSILON_SILICON;
}

// Maximum field at junction (convenience function that calculates xn internally)
float calculateMaxDepletionField(float T, float Na, float Nd, float V_applied) {
    float xp, xn;
    calculateDepletionWidths(T, Na, Nd, V_applied, xp, xn);
    return calculateMaxDepletionFieldFromXn(Nd, xn);
}

// SRH recombination rate
// R_SRH = (np - ni²) / (τ_p(n + ni) + τ_n(p + ni))
float calculateSRHRecombinationRate(float n, float p, float ni, float T) {
    float np_product = n * p;
    float ni_sq = ni * ni;

    if (np_product <= ni_sq) return 0.0f;  // No net recombination

    float numerator = np_product - ni_sq;
    float denominator = TAU_P0 * (n + ni) + TAU_N0 * (p + ni);

    return numerator / denominator;
}

// Thermal generation rate (in equilibrium, G = R)
// For non-equilibrium: net rate = R - G where G = ni² / (τ_n + τ_p) approximately
float calculateThermalGenerationRate(float ni) {
    return (ni * ni) / (TAU_N0 + TAU_P0);
}

// Diode current equation (Shockley)
// I = I_s * (exp(qV/nkT) - 1)
// I_s = A * q * ni² * (Dp/(Ln*Na) + Dn/(Lp*Nd))
float calculateDiodeCurrent(float V, float T, float Na, float Nd, float Area) {
    float ni = calculateIntrinsicConcentration(T);
    float kT_q = BOLTZMANN_EV * T;

    float mu_n = calculateMobilityWithDoping(true, T, Nd);
    float mu_p = calculateMobilityWithDoping(false, T, Na);

    // Diffusion coefficients: D = kT/q * μ
    float Dn = kT_q * mu_n;
    float Dp = kT_q * mu_p;

    // Diffusion lengths: L = sqrt(D * τ)
    float Ln = sqrt(Dn * TAU_N0);
    float Lp = sqrt(Dp * TAU_P0);

    // Saturation current
    float Is = Area * Q_ELECTRON * ni * ni * (Dp / (Ln * Na) + Dn / (Lp * Nd));

    // Ideality factor (1 for ideal diode, ~1.5-2 for real Si at low current)
    float n_ideality = 1.0f;

    // Current
    float exponent = V / (n_ideality * kT_q);
    if (exponent > 40.0f) exponent = 40.0f;  // Prevent overflow
    if (exponent < -40.0f) exponent = -40.0f;

    return Is * (exp(exponent) - 1.0f);
}

// Forward declarations
void setupDoping(int mode, vector<DopingIon>& ions, vector<Particle>& particles,
    float concentration, float T);

// Calculate local electric field at a position in the PN junction
vec2 calculateLocalField(vec2 position, float appliedV, float T,
    float Na, float Nd, int mode) {
    vec2 field = { 0.0f, 0.0f };

    // Applied field (uniform across device)
    float E_applied = appliedV / DEVICE_LENGTH_CM;
    field.x = E_applied;

    if (mode == 3) {  // PN Junction
        // Position in cm from junction (junction at x=0 in sim coords)
        float x_cm = position.x * SIM_TO_CM;

        float V_bi = calculateBuiltInPotential(T, Na, Nd);
        float V_total = V_bi - appliedV;

        if (V_total > 0.01f) {
            float xp, xn;
            calculateDepletionWidths(T, Na, Nd, appliedV, xp, xn);

            // Electric field in depletion region (linear/triangular variation)
            if (x_cm > -xp && x_cm < xn) {
                // Use optimized function - pass xn directly, no redundant calculation
                float E_max = calculateMaxDepletionFieldFromXn(Nd, xn);

                if (x_cm < 0) {
                    // P-side of depletion: field increases toward junction
                    // E(x) = E_max * (1 + x/xp) for x in [-xp, 0]
                    field.x += -E_max * (1.0f + x_cm / xp);
                }
                else {
                    // N-side of depletion: field decreases away from junction
                    // E(x) = E_max * (1 - x/xn) for x in [0, xn]
                    field.x += -E_max * (1.0f - x_cm / xn);
                }
            }
        }
    }

    return field;
}

// Setup doping configuration
void setupDoping(int mode, vector<DopingIon>& ions, vector<Particle>& particles,
    float concentration, float T) {
    ions.clear();

    // Remove free carriers from previous configuration
    particles.erase(
        remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.isFree; }),
        particles.end()
    );

    if (mode == 0) return;  // Intrinsic - no doping

    // Number of visual ions (scaled for display)
    int numVisualIons = (int)(log10(concentration) * 8);  // More at higher doping
    if (numVisualIons < 20) numVisualIons = 20;
    if (numVisualIons > 150) numVisualIons = 150;

    // Number of free carriers to show (based on relative concentrations)
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

        // Add majority carriers (electrons)
        for (int i = 0; i < numFreeCarriers; i++) {
            Particle e;
            e.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            e.homePosition = e.position;
            e.velocity = { 0, 0 };
            e.isFree = true;
            e.isElectron = true;
            e.markedForDeletion = false;
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

        // Add majority carriers (holes)
        for (int i = 0; i < numFreeCarriers; i++) {
            Particle h;
            h.position.x = ((float)rand() / RAND_MAX) * 1.7f - 0.85f;
            h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            h.homePosition = h.position;
            h.velocity = { 0, 0 };
            h.isFree = true;
            h.isElectron = false;
            h.markedForDeletion = false;
            particles.push_back(h);
        }
    }
    else if (mode == 3) {  // PN Junction
        // Calculate depletion region for initial carrier placement
        float xp, xn;
        calculateDepletionWidths(T, concentration, concentration, 0.0f, xp, xn);
        float xp_sim = xp * CM_TO_SIM;
        float xn_sim = xn * CM_TO_SIM;

        // P-side (left): acceptor ions
        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 0.85f - 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = false;
            ions.push_back(ion);
        }

        // N-side (right): donor ions
        for (int i = 0; i < numVisualIons; i++) {
            DopingIon ion;
            ion.position.x = ((float)rand() / RAND_MAX) * 0.85f;
            ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            ion.isPositive = true;
            ions.push_back(ion);
        }

        // Holes on P-side (outside depletion region)
        for (int i = 0; i < numFreeCarriers; i++) {
            Particle h;
            // Place outside depletion region on p-side
            h.position.x = -0.85f + ((float)rand() / RAND_MAX) * (0.85f - xp_sim - 0.1f);
            h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            h.homePosition = h.position;
            h.velocity = { 0, 0 };
            h.isFree = true;
            h.isElectron = false;
            h.markedForDeletion = false;
            particles.push_back(h);
        }

        // Electrons on N-side (outside depletion region)
        for (int i = 0; i < numFreeCarriers; i++) {
            Particle e;
            // Place outside depletion region on n-side
            e.position.x = xn_sim + 0.1f + ((float)rand() / RAND_MAX) * (0.85f - xn_sim - 0.1f);
            e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
            e.homePosition = e.position;
            e.velocity = { 0, 0 };
            e.isFree = true;
            e.isElectron = true;
            e.markedForDeletion = false;
            particles.push_back(e);
        }
    }
}

// Setup P-N Junction with separate Na and Nd (asymmetric doping)
void setupDopingPN(vector<DopingIon>& ions, vector<Particle>& particles,
    float Na, float Nd, float T) {
    ions.clear();

    // Remove free carriers from previous configuration
    particles.erase(
        remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.isFree; }),
        particles.end()
    );

    float ni = calculateIntrinsicConcentration(T);

    // Calculate depletion region with asymmetric doping
    float xp, xn;
    calculateDepletionWidths(T, Na, Nd, 0.0f, xp, xn);
    float xp_sim = xp * CM_TO_SIM;
    float xn_sim = xn * CM_TO_SIM;

    // Number of visual ions scaled by doping (more doping = more ions)
    int numIonsP = (int)(log10(Na) * 8);
    int numIonsN = (int)(log10(Nd) * 8);
    if (numIonsP < 20) numIonsP = 20;
    if (numIonsN < 20) numIonsN = 20;
    if (numIonsP > 150) numIonsP = 150;
    if (numIonsN > 150) numIonsN = 150;

    // Number of free carriers based on doping ratio
    float majorityRatioP = Na / ni;  // Holes on P-side
    float majorityRatioN = Nd / ni;  // Electrons on N-side
    int numHoles = (int)(log10(majorityRatioP) * 15);
    int numElectrons = (int)(log10(majorityRatioN) * 15);
    if (numHoles < 10) numHoles = 10;
    if (numElectrons < 10) numElectrons = 10;
    if (numHoles > 200) numHoles = 200;
    if (numElectrons > 200) numElectrons = 200;

    // P-side (left): acceptor ions (negative)
    for (int i = 0; i < numIonsP; i++) {
        DopingIon ion;
        ion.position.x = ((float)rand() / RAND_MAX) * 0.85f - 0.85f;
        ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        ion.isPositive = false;  // Acceptors are negative
        ions.push_back(ion);
    }

    // N-side (right): donor ions (positive)
    for (int i = 0; i < numIonsN; i++) {
        DopingIon ion;
        ion.position.x = ((float)rand() / RAND_MAX) * 0.85f;
        ion.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        ion.isPositive = true;  // Donors are positive
        ions.push_back(ion);
    }

    // Holes on P-side (outside depletion region)
    for (int i = 0; i < numHoles; i++) {
        Particle h;
        h.position.x = -0.85f + ((float)rand() / RAND_MAX) * (0.85f - xp_sim - 0.1f);
        h.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        h.homePosition = h.position;
        h.velocity = { 0, 0 };
        h.isFree = true;
        h.isElectron = false;
        h.markedForDeletion = false;
        particles.push_back(h);
    }

    // Electrons on N-side (outside depletion region)
    for (int i = 0; i < numElectrons; i++) {
        Particle e;
        e.position.x = xn_sim + 0.1f + ((float)rand() / RAND_MAX) * (0.85f - xn_sim - 0.1f);
        e.position.y = ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        e.homePosition = e.position;
        e.velocity = { 0, 0 };
        e.isFree = true;
        e.isElectron = true;
        e.markedForDeletion = false;
        particles.push_back(e);
    }
}

// Calculate density gradient and local density for diffusion
// Returns: gradient vector (∇n) and local density (n) for proper diffusion: v = -D(∇n/n)
struct DiffusionData {
    vec2 gradient;      // ∇n - concentration gradient
    float localDensity; // n - local carrier concentration
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
    data.localDensity = totalDensity + 0.1f;  // Add small value to prevent division by zero

    return data;
}

// Find recombination partner
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
            FragColor = vec4(0.2, 0.8, 1.0, 1.0);  // Cyan for free electrons
        else
            FragColor = vec4(1.0, 0.3, 0.3, 1.0);  // Red for holes
    } else {
        FragColor = vec4(0.25, 0.3, 0.4, 1.0);    // Dark blue-gray for bound
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

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
        "Semiconductor Physics Simulation - Corrected Silicon Model", NULL, NULL);
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
        ImGui::Begin("Semiconductor Lab Controls - Silicon Model");

        ImGui::Text("Device Parameters");
        ImGui::SliderFloat("Applied Voltage (V)", &appliedVoltage, -5.0f, 5.0f);
        ImGui::SliderFloat("Temperature (K)", &temperature, 200.0f, 500.0f);
        ImGui::SliderFloat("Simulation Speed", &simulationSpeed, 0.1f, 3.0f);

        if (ImGui::Button("Reset Carriers")) {
            particles.erase(
                remove_if(particles.begin(), particles.end(),
                    [](const Particle& p) { return p.isFree; }),
                particles.end()
            );
            dopingIons.clear();
            dopingMode = 0;
            currentAccumulator = 0.0f;
            currentMetric = 0.0f;
        }

        ImGui::Separator();
        ImGui::Text("Doping Configuration");

        const char* dopingModes[] = { "Intrinsic", "N-Type", "P-Type", "P-N Junction" };
        int prevDopingMode = dopingMode;
        ImGui::Combo("Device Type", &dopingMode, dopingModes, 4);

        if (dopingMode > 0 && dopingMode < 3) {
            // N-type or P-type: single doping slider
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
            // P-N Junction: separate Na and Nd sliders
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

        // PN Junction specific info
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
            ImGui::Text("  x_p = %.2e cm, x_n = %.2e cm", xp, xn);
            ImGui::Text("Max Depletion Field: %.2e V/cm", E_max);

            // Show which side has more depletion
            if (xp > xn * 1.5f) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "  -> Depletion mostly in P-side (Na < Nd)");
            }
            else if (xn > xp * 1.5f) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "  -> Depletion mostly in N-side (Nd < Na)");
            }

            // Theoretical diode current
            float Area = DEVICE_LENGTH_CM * DEVICE_LENGTH_CM;
            float I_theory = calculateDiodeCurrent(appliedVoltage, temperature,
                Na_concentration, Nd_concentration, Area);
            ImGui::Text("Theoretical I: %.3e A", I_theory);
        }

        ImGui::Separator();
        ImGui::Text("Physical Properties (Real Silicon)");

        float kT_display = BOLTZMANN_EV * temperature;
        float ni_display = calculateIntrinsicConcentration(temperature);
        float Eg = calculateBandgap(temperature);
        float mu_e = calculateMobility(true, temperature);
        float mu_h = calculateMobility(false, temperature);

        if (dopingMode > 0) {
            mu_e = calculateMobilityWithDoping(true, temperature, dopingConcentration);
            mu_h = calculateMobilityWithDoping(false, temperature, dopingConcentration);
        }

        ImGui::Text("Temperature: %.0f K (%.1f C)", temperature, temperature - 273.15f);
        ImGui::Text("Thermal Energy kT: %.4f eV", kT_display);
        ImGui::Text("Bandgap E_g: %.3f eV", Eg);
        ImGui::Text("Intrinsic n_i: %.2e cm^-3", ni_display);

        float E_field = appliedVoltage / DEVICE_LENGTH_CM;
        ImGui::Text("Applied Field: %.2e V/cm", E_field);
        ImGui::Text("Electron Mobility: %.0f cm2/(V.s)", mu_e);
        ImGui::Text("Hole Mobility: %.0f cm2/(V.s)", mu_h);

        // Diffusion coefficients
        float Dn = kT_display * mu_e;
        float Dp = kT_display * mu_h;
        ImGui::Text("D_n: %.1f cm2/s, D_p: %.1f cm2/s", Dn, Dp);

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

        ImGui::Separator();
        ImGui::Text("Current: %.2f (arb. units)", currentMetric);
        ImGui::Text("Performance: %.1f FPS", ImGui::GetIO().Framerate);

        ImGui::End();

        // ===== I-V CURVE WINDOW =====
        ImGui::Begin("I-V Characteristic");

        if (!ivSweepActive) {
            if (ImGui::Button("Start I-V Sweep")) {
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

        ImGui::SliderFloat("Step (V)", &ivSweepStep, 0.1f, 1.0f);

        // Draw I-V plot
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

            // Grid at I=0 and V=0
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

            // Plot data points
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

        // Show theoretical curve for PN junction
        if (dopingMode == 3) {
            ImGui::Separator();
            ImGui::Text("Theoretical Diode Equation:");
            ImGui::Text("I = Is * (exp(qV/kT) - 1)");

            float ni_iv = calculateIntrinsicConcentration(temperature);
            float Area = DEVICE_LENGTH_CM * DEVICE_LENGTH_CM;
            float Is = calculateDiodeCurrent(0.001f, temperature, Na_concentration,
                Nd_concentration, Area) / (exp(0.001f / (BOLTZMANN_EV * temperature)) - 1);
            ImGui::Text("Saturation current Is: %.2e A", Is);
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

        // Calculate physical parameters
        float ni = calculateIntrinsicConcentration(temperature);
        float kT_eV = BOLTZMANN_EV * temperature;

        // Count carriers
        int electronCount = 0, holeCount = 0;
        for (const Particle& p : particles) {
            if (p.isFree && !p.markedForDeletion) {
                if (p.isElectron) electronCount++;
                else holeCount++;
            }
        }

        // ===== GENERATION (SRH Model) =====
        // 
        // Real SRH Generation Rate:
        //   G = nᵢ² / [τₚ(n + nᵢ) + τₙ(p + nᵢ)]
        //
        // Physical meaning: Thermal energy creates electron-hole pairs
        // Rate depends on: temperature (via nᵢ), carrier lifetimes (τ), local concentrations
        //
        // For visualization, we use a simplified probability scaling based on nᵢ

        // Use the REAL nᵢ (already temperature-dependent)
        float ni_real = ni;

        // Simplified generation probability that scales with temperature (via nᵢ)
        // At 300K: ni ≈ 1.5e10, so (ni/NI_300K)² = 1
        // At 400K: ni ≈ 1e12, so (ni/NI_300K)² ≈ 4400
        float temp_factor = (ni_real / NI_300K);
        float generationProb = 0.00001f * temp_factor * temp_factor;
        generationProb = fmin(generationProb, 0.01f);

        bool canGenerate = (particles.size() < 5500) && (electronCount + holeCount < 400);

        for (size_t i = 0; i < particles.size() && i < (size_t)LATTICE_SIZE; i++) {
            Particle& p = particles[i];

            if (!p.isFree && p.isElectron && canGenerate) {
                if ((float)rand() / RAND_MAX < generationProb * dt) {
                    p.isFree = true;
                    p.velocity = { 0.0f, 0.0f };

                    Particle hole;
                    hole.position = p.position;
                    hole.homePosition = p.homePosition;
                    hole.velocity = { 0.0f, 0.0f };
                    hole.isFree = true;
                    hole.isElectron = false;
                    hole.markedForDeletion = false;
                    particles.push_back(hole);
                }
            }
        }

        // ===== RECOMBINATION (SRH Model) =====
        //
        // Real SRH Recombination Rate:
        //   R = (np - nᵢ²) / [τₚ(n + nᵢ) + τₙ(p + nᵢ)]
        //
        // Key physics:
        //   - Recombination only when np > nᵢ² (excess carriers above equilibrium)
        //   - (np - nᵢ²) is the "driving force" - how far from equilibrium
        //
        // For visualization, we use particle counts directly

        // Scaled ni² for particle simulation (when we have ~10 particles each, np = 100 = ni_scaled²)
        float ni_sq_scaled = 100.0f;
        float np_product = (float)(electronCount * holeCount);

        // SRH-inspired recombination: only recombine when np > ni²
        float recombProb = 0.0f;
        if (np_product > ni_sq_scaled) {
            // Rate proportional to excess carriers
            recombProb = 0.002f * (np_product - ni_sq_scaled) / (np_product + ni_sq_scaled);
        }
        recombProb = fmin(recombProb, 0.05f);

        float recombRadius = 0.15f;

        for (size_t i = 0; i < particles.size(); i++) {
            Particle& p = particles[i];
            if (!p.isFree || p.markedForDeletion || !p.isElectron) continue;

            if ((float)rand() / RAND_MAX < recombProb * dt) {
                int holeIdx = findRecombinationPartner((int)i, particles, recombRadius);

                if (holeIdx >= 0) {
                    p.isFree = false;
                    p.position = particles[holeIdx].homePosition;
                    p.homePosition = particles[holeIdx].homePosition;
                    p.velocity = { 0.0f, 0.0f };
                    particles[holeIdx].markedForDeletion = true;
                }
            }
        }

        // ===== P-N JUNCTION: MINORITY CARRIER INJECTION =====
        // In forward bias, majority carriers are injected across the junction
        // becoming minority carriers on the other side
        if (dopingMode == 3 && appliedVoltage > 0.1f) {
            float kT_q = BOLTZMANN_EV * temperature;

            // Injection probability increases exponentially with forward bias
            // P ∝ exp(V/Vt) - 1
            float exponent = appliedVoltage / (kT_q * 2.0f);  // Factor of 2 for ideality
            if (exponent > 10.0f) exponent = 10.0f;  // Prevent overflow

            float injectionProb = 0.001f * (exp(exponent) - 1.0f);
            injectionProb = fmin(injectionProb, 0.3f);  // Cap for stability

            // Inject electrons from N-side to P-side
            for (Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion || !p.isElectron) continue;
                if (p.position.x < 0.05f) continue;  // Already on or near P-side
                if (p.position.x > 0.5f) continue;   // Too far from junction

                if ((float)rand() / RAND_MAX < injectionProb * dt) {
                    // Push electron across junction (becomes minority carrier)
                    p.position.x = -0.05f - ((float)rand() / RAND_MAX) * 0.3f;
                    p.velocity.x = -0.02f;  // Moving into P-region
                }
            }

            // Inject holes from P-side to N-side
            for (Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion || p.isElectron) continue;
                if (p.position.x > -0.05f) continue;  // Already on or near N-side
                if (p.position.x < -0.5f) continue;   // Too far from junction

                if ((float)rand() / RAND_MAX < injectionProb * dt) {
                    // Push hole across junction (becomes minority carrier)
                    p.position.x = 0.05f + ((float)rand() / RAND_MAX) * 0.3f;
                    p.velocity.x = 0.02f;  // Moving into N-region
                }
            }
        }

        // ===== P-N JUNCTION: REVERSE BIAS SWEEP-OUT =====
        // In reverse bias, minority carriers near junction are swept out
        // This widens the depletion region and reduces current
        if (dopingMode == 3 && appliedVoltage < -0.1f) {
            float sweepProb = 0.01f * fabs(appliedVoltage);  // Stronger sweep at higher reverse bias
            sweepProb = fmin(sweepProb, 0.1f);

            for (Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion) continue;

                // Electrons near junction on P-side get swept to N-side
                if (p.isElectron && p.position.x > -0.3f && p.position.x < 0.0f) {
                    if ((float)rand() / RAND_MAX < sweepProb * dt) {
                        p.position.x = 0.5f + ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = 0.02f;
                    }
                }

                // Holes near junction on N-side get swept to P-side
                if (!p.isElectron && p.position.x > 0.0f && p.position.x < 0.3f) {
                    if ((float)rand() / RAND_MAX < sweepProb * dt) {
                        p.position.x = -0.5f - ((float)rand() / RAND_MAX) * 0.3f;
                        p.velocity.x = -0.02f;
                    }
                }
            }
        }

        // ===== CARRIER TRANSPORT =====

        // =====================================================
        // VELOCITY SCALING PHILOSOPHY:
        // 
        // Real physics:
        //   v_thermal ~ 10^7 cm/s (random, no net movement)
        //   v_drift ~ μE ~ 1000 cm²/Vs × 10^4 V/cm = 10^7 cm/s (at high field)
        //   v_drift ~ μE ~ 1000 cm²/Vs × 100 V/cm = 10^5 cm/s (typical)
        //
        // For visualization, we want:
        //   - Visible thermal jitter (random motion always present)
        //   - Visible drift when voltage applied (net movement)
        //   - Drift should be slower than thermal at low fields
        //   - Drift should dominate at high fields
        //
        // Screen is ~2 units wide, dt ~ 0.016s (60 FPS)
        // To cross screen in ~5 seconds at max drift: v_max ~ 0.4/5 = 0.08 units/s
        // To have visible jitter: v_thermal ~ 0.01-0.02 units/frame
        // =====================================================

        // Pre-calculate mobilities (same for all electrons / all holes)
        float doping_for_mobility = dopingMode > 0 ? dopingConcentration : 1e10f;
        float mu_electron = calculateMobilityWithDoping(true, temperature, doping_for_mobility);
        float mu_hole = calculateMobilityWithDoping(false, temperature, doping_for_mobility);

        // Mobility scaling: μ ~ 1000 cm²/Vs, we want μ_sim ~ 0.001 to get reasonable velocities
        float mu_electron_sim = mu_electron * 1e-6f;
        float mu_hole_sim = mu_hole * 1e-6f;

        // Saturation velocity scaling
        float vsat_electron_sim = VSAT_ELECTRON * 1e-7f;
        float vsat_hole_sim = VSAT_HOLE * 1e-7f;

        for (Particle& p : particles) {
            if (p.markedForDeletion) continue;

            if (p.isFree) {
                // Use pre-calculated mobility
                float mu_sim = p.isElectron ? mu_electron_sim : mu_hole_sim;
                float vsat_sim = p.isElectron ? vsat_electron_sim : vsat_hole_sim;

                // Calculate local electric field
                // Use Na/Nd for P-N junction, dopingConcentration for uniform doping
                float Na_for_field = (dopingMode == 3) ? Na_concentration : dopingConcentration;
                float Nd_for_field = (dopingMode == 3) ? Nd_concentration : dopingConcentration;
                vec2 E_field = calculateLocalField(p.position, appliedVoltage, temperature,
                    Na_for_field, Nd_for_field, dopingMode);

                // Scale field to simulation
                // E_field is in V/cm, at 5V across 0.01cm device = 500 V/cm
                // We want E_sim to give v_drift ~ 0.01-0.1 units/frame at max field
                float E_scale = 1e-4f;  // Balanced for visible but not overpowering drift
                float E_sim_x = E_field.x * E_scale;
                float E_sim_y = E_field.y * E_scale;
                float E_magnitude = sqrt(E_sim_x * E_sim_x + E_sim_y * E_sim_y);

                // Drift velocity with HIGH-FIELD CAUGHEY-THOMAS MODEL
                // v = μE / (1 + μE/vsat) - smooth saturation instead of hard clamp
                float charge = p.isElectron ? -1.0f : 1.0f;
                // vsat_sim is pre-calculated outside the loop

                float driftVelX, driftVelY;
                if (E_magnitude > 1e-10f) {
                    // Use the Caughey-Thomas function for velocity saturation
                    float v_magnitude = calculateFieldDependentVelocity(E_magnitude, p.isElectron, mu_sim, vsat_sim);

                    // Apply direction and charge sign
                    float E_norm_x = E_sim_x / E_magnitude;
                    float E_norm_y = E_sim_y / E_magnitude;
                    driftVelX = v_magnitude * E_norm_x * charge;
                    driftVelY = v_magnitude * E_norm_y * charge;
                }
                else {
                    driftVelX = 0.0f;
                    driftVelY = 0.0f;
                }

                // Diffusion: v = -D × (∇n / n)
                // Carriers move from high concentration to low concentration
                DiffusionData diffData = calculateDiffusionData(p.position, particles, p.isElectron);
                float D_sim = kT_eV * mu_sim * 0.5f;  // Einstein relation: D = (kT/q) × μ

                vec2 diffusionVel = { 0.0f, 0.0f };
                float gradientMag = length(diffData.gradient);
                if (gradientMag > 0.05f && diffData.localDensity > 0.1f) {
                    // v = -D × (∇n / n)
                    // Negative sign: move against gradient (high → low concentration)
                    diffusionVel.x = -D_sim * (diffData.gradient.x / diffData.localDensity);
                    diffusionVel.y = -D_sim * (diffData.gradient.y / diffData.localDensity);

                    // Clamp to prevent instability but allow visible spreading
                    float maxDiffVel = 0.02f;  // Reduced to not overpower thermal
                    float diffVelMag = length(diffusionVel);
                    if (diffVelMag > maxDiffVel) {
                        diffusionVel.x *= maxDiffVel / diffVelMag;
                        diffusionVel.y *= maxDiffVel / diffVelMag;
                    }
                }

                // Thermal random walk
                // This should always be visible as jittering
                // At 300K, kT_eV ~ 0.026, sqrt(0.026) ~ 0.16
                // We want thermal kick ~ 0.005-0.01 per frame for visible jitter
                float thermalSpeed = sqrt(kT_eV) * 0.05f;  // Increased for visible jitter
                float rx = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                float ry = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
                vec2 thermalKick = { rx * thermalSpeed, ry * thermalSpeed };

                // Combine velocities
                vec2 targetVel = { driftVelX + diffusionVel.x + thermalKick.x,
                                   driftVelY + diffusionVel.y + thermalKick.y };

                p.velocity += (targetVel - p.velocity) * 0.15f;
                p.position += p.velocity * dt;

                // Track current based on device type
                if (dopingMode != 3) {
                    // For resistors (intrinsic, n-type, p-type): use drift current
                    currentAccumulator += charge * driftVelX * 1000.0f;
                }
                // For P-N junction, we calculate current separately below

                // Boundary conditions
                const float WALL_L = -0.9f, WALL_R = 0.9f;
                const float WALL_B = -0.8f, WALL_T = 0.8f;

                if (p.position.x < WALL_L) { p.position.x = WALL_L; p.velocity.x *= -0.5f; }
                if (p.position.x > WALL_R) { p.position.x = WALL_R; p.velocity.x *= -0.5f; }
                if (p.position.y < WALL_B) { p.position.y = WALL_B; p.velocity.y *= -0.5f; }
                if (p.position.y > WALL_T) { p.position.y = WALL_T; p.velocity.y *= -0.5f; }

            }
            else if (p.isElectron) {
                // Bound electron thermal vibration
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

        // ===== DIODE CURRENT CALCULATION =====
        // For P-N junction, use Shockley diode equation with simulation modulation
        if (dopingMode == 3) {
            float kT_q = BOLTZMANN_EV * temperature;  // Thermal voltage in V (~0.026V at 300K)

            // Count minority carriers as a measure of injection level
            int electronsOnPside = 0;  // Minority electrons in p-region
            int holesOnNside = 0;      // Minority holes in n-region

            for (const Particle& p : particles) {
                if (!p.isFree || p.markedForDeletion) continue;
                if (p.isElectron && p.position.x < -0.1f) electronsOnPside++;
                if (!p.isElectron && p.position.x > 0.1f) holesOnNside++;
            }

            // Shockley diode equation: I = Is * (exp(V/nVt) - 1)
            float n_ideality = 1.8f;  // Ideality factor (1-2 for real diodes)
            float Is_normalized = 0.001f;  // Small saturation current

            float Vt = n_ideality * kT_q;  // ~0.047V at 300K with n=1.8
            float diodeCurrent;

            if (appliedVoltage > 0.0f) {
                // Forward bias: exponential increase
                float exponent = appliedVoltage / Vt;

                // Clamp to reasonable range (exp(10) ≈ 22000)
                if (exponent > 10.0f) exponent = 10.0f;

                diodeCurrent = Is_normalized * (exp(exponent) - 1.0f);

                // Modulate by minority carrier count (ties current to simulation state)
                // Only in forward bias where injection matters
                float minorityFactor = 1.0f + 0.02f * (electronsOnPside + holesOnNside);
                diodeCurrent *= minorityFactor;
            }
            else {
                // Reverse bias: current saturates at -Is (constant)
                // The Shockley equation gives I = Is*(exp(V/Vt) - 1)
                // For V << -Vt, exp(V/Vt) → 0, so I → -Is
                diodeCurrent = -Is_normalized;  // Flat reverse saturation current
            }

            // Scale for nice display values (target: ~0-100 range for forward, ~-1 for reverse)
            currentAccumulator = diodeCurrent * 500.0f;
        }

        // Smooth current measurement
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

            // Draw depletion region boundaries for PN junction
            if (dopingMode == 3) {
                float xp, xn;
                calculateDepletionWidths(temperature, Na_concentration,
                    Nd_concentration, appliedVoltage, xp, xn);

                float xp_sim = xp * CM_TO_SIM;
                float xn_sim = xn * CM_TO_SIM;

                // Convert to screen coordinates
                float screenXp = (-xp_sim + 1.0f) * 0.5f * windowWidth;
                float screenXn = (xn_sim + 1.0f) * 0.5f * windowWidth;
                float screenXjunction = (0.0f + 1.0f) * 0.5f * windowWidth;

                // Draw depletion region boundaries
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