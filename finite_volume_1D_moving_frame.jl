using LinearAlgebra
using Plots
using SmoothLivePlot

"""
    Uᵣₑ(t::Float64)

This function takes time as input and returns the relative velocity of the moving frame at the given time.
"""
function Uᵣₑ(t::Float64)
    return -a₀*t^2
end

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
    F[1] = ρ*(u-Uᵣₑ(t))
    F[2] = ρ*u*(u-Uᵣₑ(t)) + p
    F[3] = (u-Uᵣₑ(t))*U[3] + u*p

    return F
end

"""
    roe(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

Inert Roe's approximate Riemann solver. This function takes as input the solution and flux vectors at the left and right states, then computes an intermediate flux between two consecutive cells using Roe's approximate method.
"""
function roe(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

    # compute primitive variables
    ρₗ, uₗ, pₗ = primitive_variables(uL)
    ρᵣ, uᵣ, pᵣ = primitive_variables(uR)
    Hₗ = (uL[3] + pₗ)/ρₗ
    Hᵣ = (uR[3] + pᵣ)/ρᵣ

    # compute the Roe-average variables
    Roe_avg = [√ρₗ, √ρᵣ]/(√ρₗ + √ρᵣ)

    ũ = Roe_avg ⋅ [uₗ, uᵣ]
    H̃ = Roe_avg ⋅ [Hₗ, Hᵣ]
    Ṽ² = ũ^2
    ã = √abs((γ-1)*(H̃ - 0.5*Ṽ²))

    # compute the average eigenvalues λᵢ
    λ₁ = abs(ũ - Uᵣₑ(t))
    λ₂ = abs(ũ - ã - Uᵣₑ(t))
    λ₃ = abs(ũ + ã - Uᵣₑ(t))

    # compute the coefficients αᵢ
    Δu₁, Δu₂, Δu₃ = uR - uL

    α₁ = (γ-1)/ã^2 * ((H̃ - ũ^2)*Δu₁ + ũ*Δu₂ - Δu₃)
    α₂ = 1/(2ã) * ((ũ+ã)*Δu₁ - Δu₂ - ã*α₁)
    α₃ = Δu₁ - (α₁+α₂)

    # compute the right eigenvectors
    k̃₁ = [1.0, ũ, ũ^2/2]
    k̃₂ = [1.0, ũ-ã, H̃ - ã*ũ]
    k̃₃ = [1.0, ũ+ã, H̃ + ã*ũ]

    F = 1/2*(fL + fR) - 1/2*(α₁*λ₁*k̃₁ + α₂*λ₂*k̃₂ + α₃*λ₃*k̃₃)

    return F
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

    global t
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
    ρᵣ, uᵣ, pᵣ = primitive_variables(U[:,N])
    Uᵣ = conservative_variables([ρᵣ, 2*Uᵣₑ(t) - uᵣ, pᵣ])
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
    RK2!(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, Δt::T, h::T) where T <: Float64

This function takes as input the global array of conservative variables `U`, the left and right boundary conditions `Uₗ` and `Uᵣ`, the time step `Δt`, and the element size `h`. It then calculates `U` at time `t+Δt` using the second order Runge-Kutta method (midpoint method).
"""
function RK2!(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, Δt::T, h::T) where T <: Float64

    global t
    Uₜ = U + Δt/2 * residual(U, Uₗ, Uᵣ, h)
    t = t + Δt/2
    Uₙ = U + Δt * residual(Uₜ, Uₗ, Uᵣ, h)
    t = t + Δt/2

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
    function global_primitives(U::Array{Float64,1})

This function takes as input the global array of conservative variables `U` and returns the global array of primitive variables `W`. The array `W` is of size `nₑ*N`, where `nₑ` is the number of primitive variables at a point and `N` is the number of solution points in the whole domain. At the iᵗʰ point, we have `W[:,i] = [ρᵢ, uᵢ, pᵢ, Yᵢ]`.
"""
function global_primitives(U::Array{Float64,2})
    nₑ, N = size(U)
    W = zeros(nₑ, N)

    for i in 1:N
        W[:,i] = primitive_variables(U[:,i])
    end

    return W
end

"""
    piston(ρ₀::T, p₀::T) where T <: Float64

This function initializes the flow as a uniform field with density and pressure given by ρ₀ and p₀, and with an initial velocity equal to zero.  This is the initial state for a flow compressed by a piston.
"""
function piston(ρ₀::T, p₀::T) where T <: Float64

    global nₑ, N
    U = zeros(nₑ, N)

    U₀ = conservative_variables([ρ₀, 0.0, p₀])

    for j in 1:N
        U[:, j] = U₀
    end

    Uₗ = U₀
    Uᵣ = U₀

    return U, Uₗ, Uᵣ
end

"""
    function plot_field(mesh::Array{Float64,1}, Wₚ::Array{Float64,2}, n::Integer)

This function is a wrapper for the `plot` function of the package `Plots` made to allow the live update of plots. It takes as input the `mesh` array, the global array of primitive variables `Wₚ` (can also be used with `U`), and the number of the field that we want to plot `n`. The field numbers in `Wₚ` are the following:

* 1: density `ρ`
* 2: velocity `u`
* 3: pressure `p`
"""
function plot_field(mesh::Array{Float64,1}, Wₚ::Array{Float64,2}, n::Integer)
    plot(mesh, Wₚ[n, :], legend=false, xlims=(x₀,x₁), ylims=(0,6))
end

# simulation parameters
const nₑ = 3 # number of equations
Δt = 0.0001 # time step
t = 0.0 # time
Nₜ = 14000 # number of time iterations
Nₚ = 100 # number of iterations per plot

# generate mesh
x₀ = 0.0
x₁ = 1.0
N = 1000
mesh, h = regular_mesh(x₀, x₁, N)

# flow parameters
const γ = 1.4
ρ₀ = 1.0
p₀ = 1.0
const a₀ = √(γ*p₀/ρ₀)

# set the initial state and boundary conditions
U, Uₗ, Uᵣ = piston(ρ₀, p₀)

# generate liveplot object
Wₚ = global_primitives(U)
outPlotObject = @makeLivePlot plot_field(mesh, Wₚ, 1)

# time marching
for i in 1:Nₜ
    global U, Uₗ, Uᵣ, Δt, h, Wₚ
    U = RK2!(U, Uₗ, Uᵣ, Δt, h)
    println("iteration ", i)

    if i%Nₚ == 0
        Wₚ = global_primitives(U)
        modifyPlotObject!(outPlotObject, arg2 = Wₚ)
    end
end
