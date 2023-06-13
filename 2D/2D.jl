using LinearAlgebra, Elliptic, ClassicalOrthogonalPolynomials


function mobius(z, a, b, c, d, α)
    t₁ = a*(-α*b + b + α*c + c) - 2b*c
    t₂ = a*(α*(b+c) - b + c) - 2α*b*c
    t₃ = 2a - (α+1)*b + (α-1)*c
    t₄ = -α*(-2a+b+c) - b + c

    (t₁*z + t₂)/(t₃*z + t₄)
end


function ADI_shifts(a, b, c, d, tol)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tol)/π^2))
    α = -1 + 2γ + 2√Complex(γ^2 - γ)
    α = Real(α)

    K = Elliptic.K(1-1/α^2)
    dn = [Elliptic.Jacobi.dn((2*j + 1)*K/(2J), 1-1/α^2) for j = 0:J-1]

    [mobius(-α*i, a, b, c, d, α) for i = dn], [mobius(α*i, a, b, c, d, α) for i = dn]
end


function ADI(A, B, F, a, b, c, d, tol)
    "ADI method for solving standard sylvester AX - XB = F"
    n = size(A)[1]
    X = zeros((n, n))

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tol)/π^2))
    p, q = ADI_shifts(a, b, c, d, tol)

    for j = 1:J
        X = (F - (A - p[j]*I)*X)/(B - p[j]*I)
        X = (A - q[j]*I)\(F - X*(B - q[j]*I))
    end
    
    X
end


function ADI_generalised(A, B, C, D, F, a, b, c, d, tol)
    "ADI method for solving generalised sylvester AXB - CXD = F"
    n = size(A)[1]
    X = zeros((n, n))

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tol)/π^2))
    p, q = ADI_shifts(a, b, c, d, tol)

    for j = 1:J
        X = -(cholesky(-1*C)\(F - (A - p[j]*C)*X*B)/cholesky((D - p[j]*B)))  # C = -M₁ which is negative definite so to perform cholesky -1 is factored out
        X = cholesky(A - q[j]*C)\(F - C*X*(D - q[j]*B))/cholesky(B)
    end
    
    X
end

"Interpolate f into piecewise Legendre with n cells and degree p"
function interpolate_2D(f, n, p)
    dx = 2/n

    P₀ = legendre(0..dx)  # Legendre mapped to the reference cell
    z,T = ClassicalOrthogonalPolynomials.plan_grid_transform(P₀, (p, p))
    F = zeros(n*p, n*p)  # initialise F

    for i = 0:n-1  # loop over cells in positive x direction
        for j = 0:n-1  # loop over cells in positive y direction
            local f_ = z -> ((x,y)= z; f((x + i*dx - 1, y + j*dx - 1)))  # define f on reference cell
            F[i+1:n:n*p, j+1:n:n*p] = T * f_.(SVector.(z, z')) # interpolate f into 2D tensor Legendre polynomials on reference cell
        end
    end

    F
end


function L2_norm_2D(f, M)
    sqrt(vec(f)'*kron(M, M)*vec(f))
end