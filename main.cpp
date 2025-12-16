#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <algorithm> // For std::min

using namespace std;

// --------------------------------------------------------
// NO-DEPENDENCY VECTOR MATH HELPERS
// (Replaces GLM so you don't need to configure libraries)
// --------------------------------------------------------
struct vec2 {
    float x, y;

    // Operator overloads for easy math
    vec2 operator+(const vec2& other) const { return { x + other.x, y + other.y }; }
    vec2 operator-(const vec2& other) const { return { x - other.x, y - other.y }; }
    vec2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
    vec2& operator+=(const vec2& other) { x += other.x; y += other.y; return *this; }
    vec2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
};

// Helper functions
float length(const vec2& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

vec2 normalize(const vec2& v) {
    float len = length(v);
    if (len == 0.0f) return { 0.0f, 0.0f };
    return { v.x / len, v.y / len };
}
// --------------------------------------------------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// Settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Physics constants
float dt = 0.01f;
const float TEMPERATURE = 1.5f;
const float DAMPING = 0.90f; // Represents scattering/collisions (Drude model)

// IV Measurement Globals
float appliedVoltage = 0.0f;
float measuredCurrent = 0.0f;
int chargeCrossingCount = 0;
float measurementTimer = 0.0f;
const float MEASUREMENT_INTERVAL = 100.0f; // Time steps before logging a data point
bool sweepComplete = false;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform vec3 uColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(uColor, 1.0f);\n"
"}\n\0";

struct Particle {
    vec2 position;
    vec2 velocity;
    float mass;
    float charge; // -1 for electron, +1 for hole

    // O(1) Update logic per particle
    void update(float electricField, float dt) {
        // 1. Calculate Electric Force: F = q * E
        // Note: Voltage is potential difference. E = V / Length.
        float forceX = (charge * electricField);

        // 2. Acceleration: a = F / m
        vec2 acceleration = { forceX / mass, 0.0f };

        // 3. Update Velocity with Damping (Scattering)
        // v = v + a*dt
        velocity += acceleration * dt;

        // Apply Scattering (Energy loss to lattice)
        velocity *= DAMPING;

        // 4. Add Brownian Motion (Thermal Noise)
        // Generate random float between -1.0 and 1.0
        float randX = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        float randY = ((float)rand() / RAND_MAX * 2.0f - 1.0f);

        vec2 randomDir = { randX, randY };
        vec2 thermalNoise = normalize(randomDir) * (TEMPERATURE * 0.05f);

        velocity += thermalNoise;

        // 5. Update Position
        position += velocity * dt;

        // 6. Confine Y axis (simulate wire walls)
        if (position.y > 0.9f) { position.y = 0.9f; velocity.y *= -1; }
        if (position.y < -0.9f) { position.y = -0.9f; velocity.y *= -1; }
    }
};

int main()
{
    // Initialize Random Seed
    srand(static_cast<unsigned int>(time(0)));

    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "IV Characteristic Simulation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Compile Shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // ------------------------------------------------------------------
    // SIMULATION SETUP
    // ------------------------------------------------------------------

    const int PARTICLE_COUNT = 1000;
    vector<Particle> particles;
    particles.reserve(PARTICLE_COUNT);

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        float rx = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        float ry = ((float)rand() / RAND_MAX * 1.8f - 0.9f);
        // Treat as test charges (+1) flowing Left->Right with +Voltage
        particles.push_back({ {rx, ry}, {0.0f, 0.0f}, 1.0f, 1.0f });
    }

    // Setup VAO/VBO once
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Allocate buffer size
    glBufferData(GL_ARRAY_BUFFER, PARTICLE_COUNT * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // CSV Header for Output
    std::cout << "--- IV CHARACTERISTIC DATA ---" << endl;
    std::cout << "Voltage(V), Current(I)" << endl;

    // Render Loop
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // ------------------------------------
        // PHYSICS ENGINE (O(N) Complexity)
        // ------------------------------------

        // 1. IV Sweep Logic
        if (!sweepComplete) {
            measurementTimer += 1.0f;
            if (measurementTimer > MEASUREMENT_INTERVAL) {
                // Calculate average current over the interval
                // Current = dQ / dt. Here, count / interval.
                float current = (float)chargeCrossingCount / MEASUREMENT_INTERVAL;

                // Print Data Point
                std::cout << std::fixed << std::setprecision(2) << appliedVoltage << ", " << current * 10.0f << endl;

                // Step Voltage
                appliedVoltage += 0.1f;

                // Reset counters
                measurementTimer = 0.0f;
                chargeCrossingCount = 0;

                if (appliedVoltage > 5.0f) {
                    sweepComplete = true;
                    std::cout << "--- SWEEP COMPLETE ---" << endl;
                }
            }
        }

        // 2. Particle Update Loop
        // E = V / L. Our NDC is width 2.0 (-1 to 1).
        float electricField = (appliedVoltage * 0.5f);

        vector<float> gpuPositions;
        gpuPositions.reserve(PARTICLE_COUNT * 3);

        for (Particle& p : particles) {
            p.update(electricField, dt);

            // BOUNDARY CONDITION (Periodic / Circuit Loop)
            // If particle hits the right edge, it flows out to the "ammeter"
            // and re-enters on the left.
            if (p.position.x > 1.0f) {
                p.position.x = -1.0f; // Wrap around
                chargeCrossingCount++; // Register flow of charge
            }
            else if (p.position.x < -1.0f) {
                p.position.x = 1.0f;
            }

            gpuPositions.push_back(p.position.x);
            gpuPositions.push_back(p.position.y);
            gpuPositions.push_back(0.0f);
        }

        // ------------------------------------
        // RENDER
        // ------------------------------------

        // Update GPU memory
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpuPositions.size() * sizeof(float), gpuPositions.data());

        glUseProgram(shaderProgram);

        // Change color based on Voltage (Cold -> Hot)
        int colorLoc = glGetUniformLocation(shaderProgram, "uColor");
        float r = std::min(appliedVoltage / 5.0f, 1.0f);
        float g = 0.5f;
        float b = 1.0f - r;
        glUniform3f(colorLoc, r, g, b);

        glBindVertexArray(VAO);
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, particles.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}