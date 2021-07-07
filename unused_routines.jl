function roe_entropy_fix(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

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

    # compute the average eigenvalues λ[i]
    λ = zeros(4)
    λ[1] = ũ
    λ[2] = ũ - ã
    λ[3] = ũ + ã
    λ[4] = ũ

    # compute the left state eigenvalues
    λₗ = zeros(4)
    λₗ[1] = uₗ
    λₗ[2] = uₗ - √(γ*pₗ/ρₗ)
    λₗ[3] = uₗ + √(γ*pₗ/ρₗ)
    λₗ[4] = uₗ

    # compute the right state eigenvalues
    λᵣ = zeros(4)
    λᵣ[1] = uᵣ
    λᵣ[2] = uᵣ - √(γ*pᵣ/ρᵣ)
    λᵣ[3] = uᵣ + √(γ*pᵣ/ρᵣ)
    λᵣ[4] = uᵣ

    # compute δ and q
    δ = zeros(4)
    for i in 1:4
        δ[i] = max(0, λ[i] - λₗ[i], λᵣ[i] - λ[i])
    end

    q = zeros(4)
    for i in 1:4
        if abs(λ[i]) < δ[i]
            q[i] = δ[i]
        else
            q[i] = abs(λ[i])
        end
    end

    # compute the coefficients αᵢ
    Δu₁, Δu₂, Δu₃, Δu₄ = uR - uL

    α₁ = (γ-1)/ã^2 * ((H̃ - ũ^2)*Δu₁ + ũ*Δu₂ - Δu₃)
    α₂ = 1/(2ã) * ((ũ+ã)*Δu₁ - Δu₂ - ã*α₁)
    α₃ = Δu₁ - (α₁+α₂)
    α₄ = Δu₄ - (α₂+α₃)*Ỹ

    # compute the right eigenvectors
    k̃₁ = [1.0, ũ, ũ^2/2, 0.0]
    k̃₂ = [1.0, ũ-ã, H̃ - ã*ũ, Ỹ]
    k̃₃ = [1.0, ũ+ã, H̃ + ã*ũ, Ỹ]
    k̃₄ = [0.0, 0.0, 0.0, 1.0]

    # compute the intermediate flux vector
    F = 1/2*(fL + fR) - 1/2*(α₁*q[1]*k̃₁ + α₂*q[2]*k̃₂ + α₃*q[3]*k̃₃ + α₄*q[4]*k̃₄)

    return F
end

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
    roe(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

Inert Roe's approximate Riemann solver. This function takes as input the solution and flux vectors at the left and right states, then computes an intermediate flux between two consecutive cells using Roe's approximate method.

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
Inert Roe solver with entropy fix
"""
function roe_entropy_fix(uL::T, uR::T, fL::T, fR::T) where T <: Array{Float64,1}

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
    λ = zeros(3)
    λ[1] = ũ
    λ[2] = ũ - ã
    λ[3] = ũ + ã


    # compute the left state eigenvalues
    λₗ = zeros(3)
    λₗ[1] = uₗ
    λₗ[2] = uₗ - √(γ*pₗ/ρₗ)
    λₗ[3] = uₗ + √(γ*pₗ/ρₗ)

    # compute the right state eigenvalues
    λᵣ = zeros(3)
    λᵣ[1] = uᵣ
    λᵣ[2] = uᵣ - √(γ*pᵣ/ρᵣ)
    λᵣ[3] = uᵣ + √(γ*pᵣ/ρᵣ)

    # compute δ and q
    δ = zeros(3)
    for i in 1:3
        δ[i] = max(0, λ[i] - λₗ[i], λᵣ[i] - λ[i])
    end

    q = zeros(3)
    for i in 1:3
        if abs(λ[i]) < δ[i]
            q[i] = δ[i]
        else
            q[i] = abs(λ[i])
        end
    end

    # compute the coefficients αᵢ
    Δu₁, Δu₂, Δu₃ = uR - uL

    α₁ = (γ-1)/ã^2 * ((H̃ - ũ^2)*Δu₁ + ũ*Δu₂ - Δu₃)
    α₂ = 1/(2ã) * ((ũ+ã)*Δu₁ - Δu₂ - ã*α₁)
    α₃ = Δu₁ - (α₁+α₂)

    # compute the right eigenvectors
    k̃₁ = [1.0, ũ, ũ^2/2]
    k̃₂ = [1.0, ũ-ã, H̃ - ã*ũ]
    k̃₃ = [1.0, ũ+ã, H̃ + ã*ũ]

    # F[4] = 1/2 * (fL[4] + fR[4]) - 1/2 * (α₂*λ₂*Ỹ + α₃*λ₃*Ỹ + α₄*λ₄)
    F = 1/2*(fL + fR) - 1/2*(α₁*q[1]*k̃₁ + α₂*q[2]*k̃₂ + α₃*q[3]*k̃₃)

    return F
end
