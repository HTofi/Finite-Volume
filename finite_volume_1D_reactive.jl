using LinearAlgebra
using DelimitedFiles
using Plots
using SmoothLivePlot

"""
    primitive_variables(U::Array{Float64,1})

This function takes in the vector of conservative variables `U = [ρ, ρu, ρE, ρY]` and calculates the corresponding vector of primitive variables `W = [ρ, u, p, Y]`.

The calculation of `p` makes use of the ideal gas law.
"""
function primitive_variables(U::Array{Float64,1})

    global γ
    W::Array{Float64,1} = zeros(4)

    W[1] = U[1]
    W[2] = U[2]/U[1]
    W[3] = (γ-1) * (U[3] - 0.5*U[1]*W[2]^2)
    W[4] = U[4]/U[1]

    return W
end

"""
    conservative_variables(W::Array{Float64,1})

This function takes in the vector of primitive variables `W = [ρ, u, p, Y]` and calculates the corresponding vector of conservative variables `U = [ρ, ρu, ρE, ρY]`.

The calculation of `ρE` makes use of the ideal gas law.
"""
function conservative_variables(W::Array{Float64,1})

    U::Array{Float64,1} = zeros(4)

    U[1] = W[1]
    U[2] = W[1]*W[2]
    U[3] = W[3]/(γ-1) + 0.5*W[1]*W[2]^2
    U[4] = W[1]*W[4]

    return U
end

"""
    local_flux(U::Array{Float64, 1})

This function takes in the vector of conservative variables `U = [ρ, ρu, ρE, ρY]` at a point and calculates the flux vector `F = [ρu, ρu²+p, u(ρE+p), ρuY]` at the same point.
"""
function local_flux(U::Array{Float64,1})

    # calculate primitive variables
    ρ, u, p, Y = primitive_variables(U)

    # calculate the flux vector
    F = zeros(4)
    F[1] = ρ*u
    F[2] = ρ*u^2 + p
    F[3] = u*(U[3] + p)
    F[4] = ρ*u*Y

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

    F = zeros(4)
    for m = 1:3
        F[m] = 0.5*(fR[m]+fL[m]) - dF[m]
    end

    ## Chemistry ##

    # compute primitive variables
    ρₗ, uₗ, pₗ, Yₗ = primitive_variables(uL)
    ρᵣ, uᵣ, pᵣ, Yᵣ = primitive_variables(uR)
    Hₗ = (uL[3] + pₗ)/ρₗ
    Hᵣ = (uR[3] + pᵣ)/ρᵣ

    # compute the Roe-average variables
    Roe_avg = [√ρₗ, √ρᵣ]/(√ρₗ + √ρᵣ)

    ũ = Roe_avg ⋅ [uₗ, uᵣ]
    H̃ = Roe_avg ⋅ [Hₗ, Hᵣ]
    Ṽ² = ũ^2
    ã = √abs((γ-1)*(H̃ - 0.5*Ṽ²))
    Ỹ = Roe_avg ⋅ [Yₗ, Yᵣ]

    # compute the average eigenvalues λᵢ
    λ₁ = abs(ũ)
    λ₂ = abs(ũ - ã)
    λ₃ = abs(ũ + ã)
    λ₄ = abs(ũ)

    # compute the coefficients αᵢ
    Δu₁, Δu₂, Δu₃, Δu₄ = uR - uL

    α₁ = (γ-1)/ã^2 * ((H̃ - ũ^2)*Δu₁ + ũ*Δu₂ - Δu₃)
    α₂ = 1/(2ã) * ((ũ+ã)*Δu₁ - Δu₂ - ã*α₁)
    α₃ = Δu₁ - (α₁+α₂)
    α₄ = Δu₄ - (α₂+α₃)*Ỹ

    F[4] = 1/2 * (fL[4] + fR[4]) - 1/2 * (α₂*λ₂*Ỹ + α₃*λ₃*Ỹ + α₄*λ₄)

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
    W(Y::Float64, T::Float64)

This function takes as input the value of the progress variable at a point `Y` and the temperature at the same point `T`. It then calculates the reaction progress rate `W(Y,T)` given by Arrhenius law.
"""
function W(Y::Float64, T::Float64)
    Y = clamp(Y, 0, 1)
    if T > Tc
        return B*(1.0-Y)^ν * exp(-Eₐ/(R*T))
    else
        return 0.
    end
end

"""
    source_vector(U::Float64)

This function takes as input the vector of conservative variables at a point `U`, and returns the vector of source terms at the same point. It uses the function `W` for the reaction rate.
"""
function source_vector(U::Array{Float64,1})

    global R, qₘ
    ρ, _, p, Y = primitive_variables(U)
    T = p/(ρ*R)

    S = zeros(4)
    S[3] = ρ*qₘ*W(Y, T)
    S[4] = ρ*W(Y, T)

    return S
end

"""
    residual(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, h::T) where T <: Float64

This function takes as input the global solution vector `U`, the left and right boundary conditions `Uₗ` and `Uᵣ` and the element size `h`. It then calculates the right-hand side of Euler equations needed for time stepping.
"""
function residual(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, h::T) where T <: Float64

    F = global_flux(U, Uₗ, Uᵣ)
    rhs = -flux_divergence(F, h) + mapslices(source_vector, U, dims=[1])

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
    CJ_detonation(ρ₀::T, p₀::T, xₛ::T, mesh::Array{T,1}) where T <: Float64

This function initializes the flow field and sets the boundary conditions for the simulation of a CJ detonation propagating from right to left. The input arguments are the following:

* the upstream flow density and pressure `ρ₀` and `p₀` respectively
* the initial shock position in the comuptational domain `xₛ`
* the `mesh` vector containing the positions of all solution points

The function outputs a tuple `(U, Uₗ, Uᵣ)`. `U` is the solution array of size `nₑ*N`, where `N` is the number of elements in the mesh, and the `nₑ` correponds to the number of conservative variables at each point `[ρ, ρu, ρE, ρY]`. `Uₗ` and `Uᵣ` are respectively the left and right boundary values for the solution.

The initial state corresponds to a heaviside function with a jump in the flow variables at position `xₛ` given by the Rankine-Hugoniot relationships for an inert shock at the same Mach number. The downstream boundary condition corresponds to the sonic flow at the CJ point.
"""
function CJ_detonation(ρ₀::T, p₀::T, xₛ::T, mesh::Array{T,1}) where T <: Float64

    global nₑ, N, qₘ
    U = zeros(nₑ, N)

    # calculate Mach number of CJ detonation
    T₀ = p₀/(ρ₀*R)
    Q = (γ+1)/2 * qₘ * (γ-1)/(γ*R*T₀)
    M = √Q + √(Q+1)
    u₀ = M*√(γ*R*T₀)

    # Neumann state variables
    ρₙ = (γ+1)*M^2 / (2 + (γ-1)*M^2) * ρ₀
    uₙ = ρ₀/ρₙ * u₀
    pₙ = (1 + 2*γ/(γ+1) * (M^2 - 1)) * p₀

    U₀ = conservative_variables([ρ₀, u₀, p₀, 0.])
    Uₙ = conservative_variables([ρₙ, uₙ, pₙ, 0.])

    for j in 1:N
        if mesh[j] < xₛ
            U[:, j] = U₀
        else
            U[:, j] = Uₙ
        end
    end

    # sonic outlet at CJ point
    ρ₁ = (γ+1)*M^2 / (1 + γ*M^2) * ρ₀
    u₁ = ρ₀/ρ₁ * u₀
    p₁ = (1 + γ*M^2)/(γ+1) * p₀

    Uₗ = U₀
    Uᵣ = conservative_variables([ρ₁, u₁, p₁, 1.])

    return U, Uₗ, Uᵣ
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
    function save_data(U::Array{T,2}, path::String) where T <: Float64

This function writes the flow field data into a text file. It takes as input the global array of conservative variables `U` and the path to theoutput file `path`. It then calculates the global array of primitive variables `W` and writes it into the file given by `path`.
"""
function save_data(U::Array{T,2}, path::String) where T <: Float64
    W = global_primitives(U)

    open(path, "w") do io
        write(io, "ρ   u   p   Y\n")
        writedlm(io, W')
    end
end

"""
    function save_checkpoint(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, path::String) where T <: Float64

This function saves a snapshot of the flow field and boundary conditions at a given instant in time in order use them later for another simulation. It takes as input the global array of conservative variables `U`, the left and right boundary conditions `Uₗ` and `Uᵣ` respectively, and the path to the output file `path`. It then writes the three arrays `U`, `Uₗ` and `Uᵣ` into the file at `path`.
"""
function save_checkpoint(U::Array{T,2}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, path::String) where T <: Float64
    open(path, "w") do io
        writedlm(io, [U, Uₗ, Uᵣ])
        # writedlm(io, U)
    end
end

"""
    function load_checkpoint(path::String)

This function loads the flow field data and boundary conditions from a file that was written by `save_checkpoint`. The function takes as input the path to the file `path`. It then reads the file and outputs the three arrays `U`, `Uₗ` and `Uᵣ` that can be used to launch a new simulation.
"""
function load_checkpoint(path::String)
    io = open(path, "r")
    S = split(readline(io))
    Sₗ = split(readline(io))
    Sᵣ = split(readline(io))
    close(io)

    nₑ = length(Sₗ)
    L = length(S)
    U = zeros(L)
    Uₗ = zeros(nₑ)
    Uᵣ = zeros(nₑ)

    for i in 1:L
        U[i] = parse(Float64, S[i])
    end

    for i in 1:nₑ
        Uₗ[i] = parse(Float64, Sₗ[i])
        Uᵣ[i] = parse(Float64, Sᵣ[i])
    end

    U = reshape(U, (nₑ, L÷nₑ))

    return U, Uₗ, Uᵣ
end

"""
    function plot_field(mesh::Array{Float64,1}, Wₚ::Array{Float64,2}, n::Integer)

This function is a wrapper for the `plot` function of the package `Plots` made to allow the live update of plots. It takes as input the `mesh` array, the global array of primitive variables `Wₚ` (can also be used with `U`), and the number of the field that we want to plot `n`. The field numbers in `Wₚ` are the following:

* 1: density `ρ`
* 2: velocity `u`
* 3: pressure `p`
* 4: progress variable `Y`
"""
function plot_field(mesh::Array{Float64,1}, Wₚ::Array{Float64,2}, n::Integer)
    plot(mesh, Wₚ[n, :], legend=false, xlims=(x₀,x₁), ylims=(0,40))
end

# simulation parameters
nₑ = 4 # number of equations
Δt = 0.0002 # time step
Nₜ = 20000 # number of time iterations
Nₚ = 100 # number of iterations per plot

# generate mesh
x₀ = 0.0
x₁ = 1.0
N = 600
mesh, h = regular_mesh(x₀, x₁, N)

# flow parameters
Eₐ = 36.0
B = 10000.0
R = 28.8
qₘ = 17.5
ν = 1

γ = 1.4
ρ₀ = 1.0
p₀ = 1.0
T₀ = p₀/(ρ₀*R)
Tc = 2.0T₀
xₛ = 0.85

# set the initial state and boundary conditions
U, Uₗ, Uᵣ = CJ_detonation(ρ₀, p₀, xₛ, mesh)

# generate liveplot object
Wₚ = global_primitives(U)
outPlotObject = @makeLivePlot plot_field(mesh, Wₚ, 3)

# time marching
for i in 1:Nₜ
    global U, Uₗ, Uᵣ, Δt, h, Wₚ
    U = RK2(U, Uₗ, Uᵣ, Δt, h)
    println("iteration ", i)

    if i%Nₚ == 0
        Wₚ = global_primitives(U)
        modifyPlotObject!(outPlotObject, arg2 = Wₚ)
    end
end
