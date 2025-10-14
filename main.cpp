#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <iostream>
#include <vector>

#include <iostream>
using namespace glm;
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float dt = 0.001f; // This should ideally be calculated based on frame time

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

struct Particle {
    vec2 position, velocity;
    double mass, charge, radius;

    void applyForce(const vector<Particle>& particles, float dt) {// dt is the small incremental change between last frame and now
		vec2 totalForce(0.0f); // keep track of the total force applied to this particle
        for (const Particle& particle : particles) {
			if (&particle == this) continue; // skip self
			    // Coloumb's law: F = k * |q1 * q2| / r^2
                // r: distance between the two charges
			vec2 difference = particle.position - position;
			float distance = length(difference);
			//distance -= particle.radius + radius; // consider the radius of the particles

			vec2 direction = normalize(difference); // direction from this particle to the other
            float softened = sqrt(distance * distance + 0.01f * 0.01f);
			float force_magnitude = (particle.charge * charge) / (softened * softened); // k is assumed to be 1 for simplicity
            // positive force magnitude would be a force vector TOWARDS the particle
            // psitive force is obtained with similarly charged particles, hence invert the force magnitude
			totalForce += -force_magnitude * direction; // accumulate the forces from all other particles
            
        }

		//update velocity and position based on the total force
		vec2 acceleration = totalForce / (float)mass; // a = F / m
		velocity += acceleration * dt; // v = u + at
		position += velocity * dt; // s = s0 + vt
    }

	//Add brownian motion to the particle based on its temperature
    void addBrownianMotion() {
		//Maxwell Boltzmann relaiton for 2D simualtions: V = sqrt(2 * k * T / m)
		float k = 1.0f; // Boltzmann constant
		float temperature = 1.0f; // room temperature in Kelvin
		float thermalVelcoity = sqrt(2 * k * temperature / (float)mass);

		//Generate a random direction
        float randX = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        float randY = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        vec2 randomDir = normalize(vec2(randX, randY));

        float damping = 0.98f; // drag coefficient
		//Langevin dynamics (v (t+1) = v(t) * damping + randomForce)
        velocity = (velocity * damping) + (randomDir * thermalVelcoity * 0.1f); // scale down for stability
    }

        //Add electron drift 
        void addDrift(int numElectrons) {
            float n = (float)numElectrons / 1.0f; // assume 1 unit area

            // target drift velocity to the right
            vec2 drift = vec2(0.5f / n, 0.0f);
            velocity = mix(velocity, drift, 0.05f); // gently steer toward drift velocity
        }

    //Generate electron-hole pairs (based on rate which is determine by light intensity)
    static void carrierGeneration(vector <Particle>& particles, float generationRate, float dt) {
        if (((float)rand() / RAND_MAX) < generationRate * dt) {
			//pick a random position near the current particle
            float randX = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            float randY = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
            vec2 position(randX, randY);

            Particle e = { position + vec2(0.02f, 0.0f), vec2(0.0f), 5.0, -1, 0.02 };

			particles.push_back(e);
        }
	}

	//Recombination of electron-hole pairs (based on proximity)
    static void carrierRecombination(vector <Particle>& particles, float recombinationRate) {
        //Randomly remove one of the particles (this depends on if the electron dropps an energy level and recombines with a hole- which is a probabilistic behaviour)
        //For simplicity, we will randomly 1. decide wether or not to remove a particle 2.pick a random particle to remove
        for (int i = 0; i < particles.size(); ++i) {
            float chance = ((float)rand() / RAND_MAX);
            if (chance < recombinationRate * dt) {
                particles.erase(particles.begin() + i);
                --i;
            }
        }
    }
};
int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    vector<Particle> particles = {
        { vec2(-0.3f, 0.0f), vec2(0.0f), 5.0, -1.0, 0.02 },  // left, positive
        //{ vec2(0.3f, 0.0f), vec2(0.0f), 5.0, 1.0, 0.02 },  // right, negative
        //{ vec2(0.0f, 0.3f), vec2(0.0f), 5.0, 0.2, 0.02 }
    };

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.125f, 0.141f, 0.141f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

		//carrier generation rate (based on light intensity)
        Particle::carrierGeneration(particles, 1.0f, dt); // 5 pairs/sec
		int n = particles.size();

        //Apply forces and update particle positions
        for (Particle& particle : particles) {
            particle.applyForce(particles, dt);
            particle.addBrownianMotion();
            particle.addDrift(n);
            
		}

		Particle::carrierRecombination(particles, 0.1f); // recombination radius

        // collect particle positions into a flat array
        vector<float> positions;
        for (auto& p : particles) {
            positions.push_back(p.position.x);
            positions.push_back(p.position.y);
            positions.push_back(0.0f);
        }

        // create VAO/VBO for the particles
        unsigned int VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // draw particles as squares (GL_POINTS)
        glUseProgram(shaderProgram);
        glPointSize(20.0f);
        glDrawArrays(GL_POINTS, 0, particles.size());

        // cleanup
        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &VAO);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}