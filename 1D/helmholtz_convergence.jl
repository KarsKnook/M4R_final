using PiecewiseOrthogonalPolynomials, Plots, LinearAlgebra, LaTeXStrings
import PiecewiseOrthogonalPolynomials: ∞

include("1D.jl")

"
Solving k²<u,v> + <∇u, ∇v> = <f,v> on [-1, 1] using optimised Cholesky decomposition
"

k = 1
n = 4  # amount of cells

r = range(-1, 1; length=n+1)
C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2)P_k^(1,1)
x = axes(C,1)
D = Derivative(x)
M = C'*C  # mass matrix
D₂ = (D*C)'*(D*C)  # weak Laplacian

# homogeneous Dirichlet boundary conditions
#u_exact = sin.(pi*x); f = k^2*u_exact .+ pi^2*sin.(pi*x)
#u_exact = (1 .- x.^2).*exp.(x); f = k^2*u_exact .+ exp.(x).*(x.^2 .+ 4x .+ 1)
θ = 5; u_exact = sin.(θ*pi*x.^2); f = k^2*u_exact .- 2*θ*pi .*(cos.(θ*pi*x.^2) .- 2*θ*pi*x.^2 .*sin.(θ*pi*x.^2))

# homogeneous Neumann boundary conditions
#f = 25 .- 25sin.(pi*x).^2 .+ 2pi^2*(cos.(pi*x).^2 .- sin.(pi*x).^2); u_exact = 1 .- sin.(pi*x).^2
#f = 25exp.(x).*(1 .- x).^2 .*(1 .+ x).^2 .- exp.(x).*(x.^4 .+ 8x.^3 .+ 10x.^2 .- 8x .- 3); u_exact = (1 .- x).^2 .* (1 .+ x).^2 .* exp.(x)

p_values = 1:15
L2 = zeros(length(p_values))

for (index, p) = enumerate(p_values)
    A = k^2*M[Block.(1:p+2),Block.(1:p+2)] + D₂[Block.(1:p+2),Block.(1:p+2)]  # up to W_p
    A = construct_Helmholtz_matrix(n, p, A, "dirichlet")

    b = construct_b(n, p, C, M, f, "dirichlet")
    b = b[end:-1:1]  # reversing RHS vector

    u = cholesky(A) \ b  # solve
    u = u[end:-1:1]  # reversing coefficient vector
    u = C*[u; zeros(∞)]  # construct solution

    #L2 = L2_norm_1D(C \ u - C \ u_exact, M) # is what we want but this computation hangs
    L2[index] = sqrt(L2_norm_1D(C\u_exact, M)^2 + L2_norm_1D(C\u, M)^2 - 2*(C\u)'*M*(C\u_exact))
end

plot(p_values, L2, yaxis=:log10, xlabel=L"p", ylabel=L"||u - u_{exact}||_{L^2}", legend = false)