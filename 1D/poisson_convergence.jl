using PiecewiseOrthogonalPolynomials, Plots, LinearAlgebra, LaTeXStrings
import PiecewiseOrthogonalPolynomials: ∞

include("1D.jl")

"
Solving k²<u,v> + <∇u, ∇v> = <f,v> on [-1, 1] using optimised Cholesky decomposition
"

n = 3  # amount of cells

r = range(-1, 1; length=n+1)
C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2)P_k^(1,1)
x = axes(C,1)
D = Derivative(x)
M = C'*C  # mass matrix
D₂ = (D*C)'*(D*C)  # weak Laplacian

# homogeneous Dirichlet boundary conditions
#f = pi^2*sin.(pi*x); u_exact = sin.(pi*x)
#f = exp.(x).*(x.^2 .+ 4x .+ 1); u_exact = (1 .- x.^2).* exp.(x)
θ = 4; u_exact = sin.(θ*pi*x).*exp.(x); f = -exp.(x).*((1-pi^2*θ^2)*sin.(θ*pi*x) .+ (2*pi*θ)*cos.(θ*pi*x))

p_values = 1:10
L2 = zeros(length(p_values))

for (index, p) = enumerate(p_values)
    A = construct_Poisson_matrix(n, p, D₂, "dirichlet") 

    b = construct_b(n, p, C, M, f, "dirichlet")

    u = A \ b  # solve
    u = C*[u; zeros(∞)]  # construct solution

    L2[index] = sqrt(L2_norm_1D(C\u_exact, M)^2 + L2_norm_1D(C\u, M)^2 - 2*(C\u)'*M*(C\u_exact))
end

plot(p_values, L2, yaxis=:log10, xlabel=L"p", ylabel=L"||u - u_{exact}||_{L^2}", legend = false)