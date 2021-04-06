"""
    primitive_variables(U::Array{Float64,1})

This function takes in the vector of conservative variables `U = [ρ, ρu, ρE]` and calculates the corresponding vector of primitive variables `W = [ρ, u, p]`.

The calculation of `p` makes use of the ideal gas law.
"""
function primitive_variables(U::Array{Float64,1})

    global γ
    W::Array{Float64,1} = zeros(3)

    W[1] = U[1]
    W[2] = U[2]/U[1]
    W[3] = (γ-1) * (U[3] - 0.5*U[1]*W[2]^2)

    return W
end

"""
    conservative_variables(W::Array{Float64,1})

This function takes in the vector of primitive variables `W = [ρ, u, p]` and calculates the corresponding vector of conservative variables `U = [ρ, ρu, ρE]`.

The calculation of `ρE` makes use of the ideal gas law.
"""
function conservative_variables(W::Array{Float64,1})

    U::Array{Float64,1} = zeros(3)

    U[1] = W[1]
    U[2] = W[1]*W[2]
    U[3] = W[3]/(γ-1) + 0.5*W[1]*W[2]^2

    return U
end

"""
    local_flux(U::Array{Float64, 1})

This function takes in the vector of conservative variables `U = [ρ, ρu, ρE]` at a point and calculates the flux vector `F = [ρu, ρu²+p, u(ρE+p)]` at the same point.
"""
function local_flux(U::Array{Float64,1})

    # calculate primitive variables
    ρ, u, p = primitive_variables(U)

    # calculate the flux vector
    F = zeros(3)
    F[1] = ρ*u
    F[2] = ρ*u^2 + p
    F[3] = u*(U[3] + p)

    return F
end

"""
    roe(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

Roe's approximate Riemann solver. This function takes as input the solution and flux vectors at the left and right states, then computes an intermediate flux between two consecutive cells using Roe's approximate method.

Source: https://github.com/surajp92/CFD_Julia
"""
function roe(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}
    dd = Array{Float64}(undef,3)
    dF = Array{Float64}(undef,3)
    V = Array{Float64}(undef,3)
    gm = γ-1.0

    #Left and right states:
    rhLL = uL[1]
    uuLL = uL[2]/rhLL
    eeLL = uL[3]/rhLL
    ppLL = gm*(eeLL*rhLL - 0.5*rhLL*(uuLL*uuLL))
    hhLL = eeLL + ppLL/rhLL

    rhRR = uR[1]
    uuRR = uR[2]/rhRR
    eeRR = uR[3]/rhRR
    ppRR = gm*(eeRR*rhRR - 0.5*rhRR*(uuRR*uuRR))
    hhRR = eeRR + ppRR/rhRR

    alpha = 1.0/(sqrt(abs(rhLL)) + sqrt(abs(rhRR)))

    uu = (sqrt(abs(rhLL))*uuLL + sqrt(abs(rhRR))*uuRR)*alpha
    hh = (sqrt(abs(rhLL))*hhLL + sqrt(abs(rhRR))*hhRR)*alpha
    aa = sqrt(abs(gm*(hh-0.5*uu*uu)))

    D11 = abs(uu)
    D22 = abs(uu + aa)
    D33 = abs(uu - aa)

    beta = 0.5/(aa*aa)
    phi2 = 0.5*gm*uu*uu

    #Right eigenvector matrix
    R11, R21, R31 = 1.0, uu, phi2/gm
    R12, R22, R32 = beta, beta*(uu + aa), beta*(hh + uu*aa)
    R13, R23, R33 = beta, beta*(uu - aa), beta*(hh - uu*aa)

    #Left eigenvector matrix
    L11, L12, L13 = 1.0-phi2/(aa*aa), gm*uu/(aa*aa), -gm/(aa*aa)
    L21, L22, L23 = phi2 - uu*aa, aa - gm*uu, gm
    L31, L32, L33 = phi2 + uu*aa, -aa - gm*uu, gm

    for m = 1:3
        V[m] = 0.5*(uR[m]-uL[m])
    end

    dd[1] = D11*(L11*V[1] + L12*V[2] + L13*V[3])
    dd[2] = D22*(L21*V[1] + L22*V[2] + L23*V[3])
    dd[3] = D33*(L31*V[1] + L32*V[2] + L33*V[3])

    dF[1] = R11*dd[1] + R12*dd[2] + R13*dd[3]
    dF[2] = R21*dd[1] + R22*dd[2] + R23*dd[3]
    dF[3] = R31*dd[1] + R32*dd[2] + R33*dd[3]

    f = zeros(3)
    for m = 1:3
        f[m] = 0.5*(fR[m]+fL[m]) - dF[m]
    end

    return f
end

"""
    flux_divergence(F::Array{Float64,3}, h::Float64)

This function takes as input the cell size `h` and the global flux array `F` of size `nₑ*2*N`, where `nₑ` is the number of equations (i.e number of conservative variables), `2` is the right and left flux nodes per element, and `N` is the number of elements in the mesh. It then calculates the global flux divergence `∇F` at solution points in the whole domain.
"""
function flux_divergence(F::Array{Float64,3}, h::Float64)

    nₑ, _, N = size(F)
    ∇F = zeros(nₑ, N)

    for j in 1:N, i in 1:nₑ
        ∇F[i,j] = (F[i,2,j] - F[i,1,j])/h
    end

    return ∇F
end

"""
    global_flux(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}) where T <: Float64

This function takes as input the global array of conservative variables `U` and the conservative variable vectors at the left and right boundaries `Uₗ` and `Uᵣ` respectively. It outputs the global flux array `F` of size `nₑ*2*N` where `nₑ` correponds to the number of conservative variables, `2` is the number of flux points per element, and `N` is the number of elements in the mesh.
"""
function global_flux(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}) where T <: Float64

    nₑ, N = size(U)

    # global flux array
    F = zeros(nₑ, 2, N)

    # calculate element-wise flux
    for j in 1:N
        F[:,1,j] = local_flux(U[:,j])
        F[:,2,j] = F[:,1,j]
    end

    # solve Riemann problem at cell interfaces
    for j in 1:N-1
        F[:,2,j] = roe(U[:,j], U[:,j+1], F[:,2,j], F[:,1,j+1])
        F[:,1,j+1] = F[:,2,j]
    end

    # calculate flux at boundaries
    F[:,1,1] = roe(Uₗ, U[:,1], local_flux(Uₗ), F[:,1,1])
    F[:,2,N] = roe(U[:,N], Uᵣ, F[:,2,N], local_flux(Uᵣ))

    return F
end

"""
    residual(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, h::T) where T <: Float64

This function takes as input the global solution vector `U`, the left and right boundary conditions `Uₗ` and `Uᵣ` and the element size `h`. It then calculates the right-hand side of Euler equations needed for time stepping.
"""
function residual(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, h::T) where T <: Float64

    F = global_flux(U, Uₗ, Uᵣ)
    rhs = -flux_divergence(F, h)

    return rhs
end

"""
    RK2(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, Δt::T, h::T) where T <: Float64

This function takes as input the global array of conservative variables `U`, the left and right boundary conditions `Uₗ` and `Uᵣ`, the time step `Δt`, and the element size `h`. It then calculates `U` at time `t+Δt` using the second order Runge-Kutta method (midpoint method).
"""
function RK2(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, Δt::T, h::T) where T <: Float64

    Uₜ = U + Δt/2 * residual(U, Uₗ, Uᵣ, h)
    Uₙ = U + Δt * residual(Uₜ, Uₗ, Uᵣ, h)

    return Uₙ
end

"""
    regular_mesh(x₀::Float64, x₁::Float64, N::Int64)

This function takes in the left and right boundaries of the computational domain `x₀` and `x₁` respectively, and the number of elements in the domain `N`. It then returns a tuple consisting of a 1D array called `mesh` containing the positions of the centers of the elements, as well as the element size `h`.
"""
function regular_mesh(x₀::Float64, x₁::Float64, N::Int64)

    mesh = zeros(N)
    xₗ = range(x₀, x₁, length = N+1)
    h = (x₁ - x₀)/N

    for j in 1:N
        mesh[j] = xₗ[j] + h/2
    end

    return mesh, h
end

"""
    shock(M::T, ρ₀::T, p₀::T, xₛ::T, Δu::T, mesh::Array{T,1}) where T <: Float64

This function initializes the flow field for the simulation of an inert shock wave. The input arguments are the following:

* the shock Mach number `M`
* the upstream flow density and pressure `ρ₀` and `p₀` respectively
* the shock position in the comuptational domain `xₛ`
* the velocity of the shock in the computional domain `Δu`
* the `mesh` vector containing the positions of all solution points

The function outputs a tuple `(U, Uₗ, Uᵣ)`. `U` is the solution array of size `nₑ*N`, where `N` is the number of elements in the mesh, and the `nₑ` correponds to the number of conservative variables at each point `[ρ, ρu, ρE]`. `Uₗ` and `Uᵣ` are respectively the left and right boundary values for the solution.

The initial state corresponds to a heaviside function with a jump in the flow variables at position `xₛ` given by the Rankine-Hugoniot relationships.
"""
function shock(M::T, ρ₀::T, p₀::T, xₛ::T, Δu::T, mesh::Array{T,1}) where T <: Float64

    global nₑ, N
    U = zeros(nₑ, N)

    u₀ = M*√(γ*p₀/ρ₀)
    ρ₁ = (γ+1)*M^2 / (2 + (γ-1)*M^2) * ρ₀
    u₁ = ρ₀/ρ₁ * u₀
    p₁ = (1 + 2*γ/(γ+1) * (M^2 - 1)) * p₀

    U₀ = conservative_variables([ρ₀, u₀+Δu, p₀])
    U₁ = conservative_variables([ρ₁, u₁+Δu, p₁])

    for j in 1:N
        if mesh[j] < xₛ
            U[:, j] = U₀
        else
            U[:, j] = U₁
        end
    end

    Uₗ = U₀
    Uᵣ = U₁

    return U, Uₗ, Uᵣ
end

"""
    sod_tube(ρ₀::T, p₀::T, ρ₁::T, p₁::T, xₛ::T, mesh::Array{T,1}) where T <: Float64

This function sets the initial and boundary conditions for the Sod tube problem. It takes as input the pressure and density in the two sides of the tube (`ρ₀`, `p₀` for the left side and `ρ₁`, `p₁` at the right) as well as the position of the initial split `xₛ` and the `mesh` vector containing the positions of solution nodes.

The function then outputs a tuple `(U, Uₗ, Uᵣ)`. `U` is the solution array of size `nₑ*N`, where `N` is the number of elements in the mesh, and the `nₑ` correponds to the number of conservative variables at each point `[ρ, ρu, ρE]`. `Uₗ` and `Uᵣ` are respectively the left and right boundary values for the solution.
"""
function sod_tube(ρ₀::T, p₀::T, ρ₁::T, p₁::T, xₛ::T, mesh::Array{T,1}) where T <: Float64

    global nₑ, N
    U = zeros(nₑ, N)

    U₀ = conservative_variables([ρ₀, 0.0, p₀])
    U₁ = conservative_variables([ρ₁, 0.0, p₁])

    for j in 1:N
        if mesh[j] < xₛ
            U[:, j] = U₀
        else
            U[:, j] = U₁
        end
    end

    Uₗ = U₀
    Uᵣ = U₁

    return U, Uₗ, Uᵣ
end

using Plots

# simulation parameters
nₑ = 3 # number of equations
Δt = 0.0001 # time step
Nₜ = 2000 # number of time iterations
Nₚ = 100 # number of iterations per plot

# generate mesh
x₀ = 0.0
x₁ = 1.0
N = 1000
mesh, h = regular_mesh(x₀, x₁, N)

# flow parameters
γ = 1.4
ρ₀ = 1.0
p₀ = 1.0
M = 2.0
xₛ = 0.5
Δu = -0.1
ρ₁ = 0.125
p₁ = 0.1

# set the initial state and boundary conditions
# U, Uₗ, Uᵣ = shock(M, ρ₀, p₀, xₛ, Δu, mesh)
U, Uₗ, Uᵣ = sod_tube(ρ₀, p₀, ρ₁, p₁, xₛ, mesh)

# time marching
for i in 1:Nₜ
    global U, Uₗ, Uᵣ, Δt, h
    U = RK2(U, Uₗ, Uᵣ, Δt, h)
    println("iteration ", i)

    if i%Nₚ == 0
        plot(mesh, U[1, :], legend=false, xlims=(0,1), ylims=(0,1.2)) |> display
    end
end
