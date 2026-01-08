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

// ---------- OPERATIONS ----------

//Operator overloading
struct vec2 {
    float x, y;
    vec2 operator+(const vec2& other) const { return { x + other.x, y + other.y }; }
    vec2 operator-(const vec2& other) const { return { x - other.x, y - other.y }; }
    vec2 operator*(float scalar) const { return { x * scalar, y * scalar }; }
    vec2& operator+=(const vec2& other) { x += other.x; y += other.y; return *this; }
    vec2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
};

// Finding the length of the vector
float length(const vec2& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

// normalizing vectors (v/|v|)
vec2 normalize(const vec2& v) {
    float len = length(v);
    if (len == 0.0f) return { 0.0f, 0.0f };
    return { v.x / len, v.y / len };
}

// ---------- GLOBAL VARIABLES ----------
const unsigned int SCR_WIDTH = 1200; // Wider for UI
const unsigned int SCR_HEIGHT = 800;

// Simulation Variables 
float appliedVoltage = 0.0f;
float temperature = 0;
float simulationSpeed = 1.0f;

// Measurement
int chargeCrossingCount = 0;
float currentMetric = 0.0f;
float measureTimer = 0.0f;

struct Particle {
    vec2 position;
    vec2 velocity;
    vec2 homePosition;
    bool isFree;

    void update(float electricField, float dt, float temp) {
        float damping = 0.90f;
        float bandGapChance = 0.002f;
        float recombinationRate = 0.01f;

        float randomVal = (float)rand() / RAND_MAX;

        // 1. STATE TRANSITIONS
        if (!isFree) {
            // Jump to Conduction Band
            if (randomVal < bandGapChance * temp) { // the probability of an electron jumping to the conduction band increases with temperature
                isFree = true;
            }
        }
        else {
            // Recombination
            if (randomVal < recombinationRate) {
                isFree = false;
                velocity = { 0.0f, 0.0f };
                homePosition = position; // falls back to a nearby hole
            }
        }

        // 2. MOVEMENT
        if (isFree) {
            float forceX = electricField;
            vec2 acceleration = { forceX, 0.0f };
            velocity += acceleration * dt;
			velocity *= damping; // electrons constantly lose energy by bumping into atoms (prevents electrons from accelerating indefinitely)

            // Thermal Jitter
            float rX = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            float rY = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            velocity += normalize({ rX, rY }) * (temp * 0.05f);

            position += velocity * dt;
        }
        else {
            // Valence Vibration
            float rX = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            float rY = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            vec2 vibration = { rX * 0.005f * temp, rY * 0.005f * temp };
			vec2 pullBack = (homePosition - position) * 0.1f; // Theres an electrostatic pull towards the home position (towards lower potential)
            position += pullBack + vibration; 
        }

        // 3. BOUNDARIES
        if (position.y > 0.9f) { position.y = 0.9f; if (isFree) velocity.y *= -1; }
        if (position.y < -0.9f) { position.y = -0.9f; if (isFree) velocity.y *= -1; }
    }
};

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in float aIsFree;\n"
"out float isFree;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   isFree = aIsFree;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"in float isFree;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   if(isFree > 0.5) \n"
"       FragColor = vec4(1.0f, 0.9f, 0.2f, 1.0f); // Yellow\n"
"   else \n"
"       FragColor = vec4(0.3f, 0.4f, 0.5f, 1.0f); // Gray\n"
"}\n\0";

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

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

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Semiconductor Physics Simulation", NULL, NULL);
    if (window == NULL) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { return -1; }

    // ------------------------------------
    // SETUP IMGUI
    // ------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark(); // Cool dark theme
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Shader Setup
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

    // Inital Particle Setup
    vector<Particle> particles;
    int rows = 30;
    int cols = 50;
	particles.reserve(rows * cols); //We will be dealing with a fixed number of particles (rows * cols)

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            float x = (float)i / cols * 1.8f - 0.9f;
            float y = (float)j / rows * 1.6f - 0.8f;
            particles.push_back({ {x, y}, {0,0}, {x,y}, false });
        }
    }

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // IMGUI FRAME START
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI DESIGN
        ImGui::Begin("Lab Controls");

        ImGui::Text("Device Parameters");
        ImGui::SliderFloat("Voltage (V)", &appliedVoltage, 0.0f, 10.0f);
        ImGui::SliderFloat("Temperature (T)", &temperature, 0.0f, 5.0f);
        ImGui::SliderFloat("Time Scale", &simulationSpeed, 0.0f, 2.0f);

        ImGui::Separator();

        ImGui::Text("Measurements");
        // Simple color change for current text
        if (currentMetric > 5.0f) ImGui::TextColored(ImVec4(1, 1, 0, 1), "Current: %.2f mA", currentMetric);
        else ImGui::Text("Current: %.2f mA", currentMetric);

        ImGui::Text("Active Carriers: %d", (int)count_if(particles.begin(), particles.end(), [](Particle& p) { return p.isFree; }));

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        // PHYSICS UPDATE
        
        // Calculate current metric
        measureTimer += 1.0f;
        if (measureTimer > 60.0f) {
            currentMetric = (float)chargeCrossingCount / 6.0f; // Scale for display
            chargeCrossingCount = 0;
            measureTimer = 0.0f;
        }

        float dt = 0.01f * simulationSpeed;
        float electricField = appliedVoltage * 0.3f;
        vector<float> gpuData;

        for (Particle& p : particles) {
            p.update(electricField, dt, temperature);

            if (p.isFree && p.position.x > 1.0f) {
                p.position.x = -1.0f;
                chargeCrossingCount++;
            }
            else if (p.isFree && p.position.x < -1.0f) {
                p.position.x = 1.0f;
            }

            gpuData.push_back(p.position.x);
            gpuData.push_back(p.position.y);
            gpuData.push_back(0.0f);
            gpuData.push_back(p.isFree ? 1.0f : 0.0f);
        }


        // RENDER
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpuData.size() * sizeof(float), gpuData.data());

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, particles.size());

        // Render ImGui over the top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup ImGui
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