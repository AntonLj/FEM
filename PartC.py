from dolfin import *

# Create mesh and define function space
mesh = Mesh ("meshes/circle.xml.gz")

# Construct the finite element space
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P1 * P1 * P1
W = FunctionSpace(mesh, TH)

# Define parameters :
T = 1000
dt = 0.5
delta1 = 1
delta2 = 1
delta3 = 1
alpha = 0.4
beta = 1
gamma = 0.8
zeta = 2
L_0 = 0.4
l = 0.4
m = 0.12

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def eval(self, values, x):
        values [1] = 4/15-2*pow(10,-7)*(x[0]-0.1*x[1]-350)*(x[0]-0.1*x[1]-67)
        values [2] = 22/45-3*pow(10,-5)*(x[0]-450)-1.2*pow(10,-4)*(x[1]-15)
        values [0] = 0.01*values[1]

    def value_shape (self):
        return (3,)

# Define initial condition
indata = InitialConditions(degree=2)
u0 = Function(W)
u0 = interpolate(indata, W)

# Test and trial function
u = TrialFunction(W)
v = TestFunction(W)

# Create bilinear and linear forms

a0 = u[0]*v[0]*dx + 1/2*dt*delta1*inner(grad(u[0]), grad(v[0]))*dx - alpha*1/2*dt*u[0]*v[0]*dx

a1 = u[1]*v[1]*dx + 1/2*dt*delta2*inner(grad(u[1]), grad(v[1]))*dx - beta*1/2*dt*u[1]*v[1]*dx

a2 = u[2]*v[2]*dx + 1/2*dt*delta3*inner(grad(u[2]), grad(v[2]))*dx + gamma*dt*1/2*u[2]*v[2]*dx

L0 = u0[0]*v[0]*dx - 1/2*dt*delta1*inner(grad(u0[0]), grad(v[0]))*dx + alpha*1/2*dt*u0[0]*v[0]*dx - dt*((alpha*u0[0]*u0[0])/(L_0 + l*u0[1]))*v[0]*dx

L1 = u0[1]*v[1]*dx - 1/2*dt*delta2*inner(grad(u0[1]), grad(v[1]))*dx + beta*1/2*dt*u0[1]*v[1]*dx - dt*((u0[1]*u0[2])/(alpha+u0[1] + m*u0[0]) + beta*u0[1]*u0[1])*v[1]*dx

L2 = u0[2]*v[2]*dx - 1/2*dt*delta3*inner(grad(u0[2]), grad(v[2]))*dx - gamma*dt*1/2*u0[2]*v[2]*dx + dt*(zeta*u0[1]*u0[2])/(alpha+u0[1] + m*u0[0])*v[2]*dx

a = a0+a1+a2
L = L0+L1+L2

# Set initial condition
u = Function(W)
u.assign(u0)

# Create population integrals
M0 = u[0]*dx
M1 = u[1]*dx
M2 = u[2]*dx

# Create population vectors
pop_0_vector = []
pop_1_vector = []
pop_2_vector = []

# Calculate initial population rate
pop_0 = assemble(M0)
pop_1 = assemble(M1)
pop_2 = assemble(M2)

# Save initial population rate
pop_0_vector.append(pop_0)
pop_1_vector.append(pop_1)
pop_2_vector.append(pop_2)

# Time - stepping
t = 0

# Save initial state
File("results/0C1b.pvd") << u

while t < T:
    # Step time
    t = t + dt
    
    # Assign u0
    u0.assign(u)
    A = assemble(a)
    b = assemble(L)

    solve(A, u.vector(), b, "lu")
    
    print(t)
    
    # Calculate population rate
    pop_0 = assemble(M0)
    pop_1 = assemble(M1)
    pop_2 = assemble(M2)
    
    # Save population rate
    pop_0_vector.append(pop_0)
    pop_1_vector.append(pop_1)
    pop_2_vector.append(pop_2)
    
    # Save solution
    if t == 100:
        File("results/100C1b.pvd") << u
    elif t == 200:
        File("results/200C1b.pvd") << u
    elif t == 300:
        File("results/300C1b.pvd") << u
    elif t == 400:
        File("results/400C1b.pvd") << u
    elif t == 1000:
        File("results/1000C1b.pvd") << u


