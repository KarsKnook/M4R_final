using PiecewiseOrthogonalPolynomials, Plots, LinearAlgebra, LaTeXStrings
import PiecewiseOrthogonalPolynomials: ∞

include("1D.jl")

"
Solving <∇u, ∇v> = <f,v> on [-1, 1] using standard \
"

n = 3  # amount of cells
p = 4  # degree, up to W_p

r = range(-1, 1; length=n+1)
C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2)P_k^(1,1)
x = axes(C,1)
D = Derivative(x)
M = C'*C  # mass matrix
D₂ = (D*C)'*(D*C)  # weak laplacian

A = D₂[Block.(1:p+2),Block.(1:p+2)]  # upto polynomial degree p
A[:,1] .= 0; A[1,:] .= 0; A[1,1] = 1.; A[:,n+1] .= 0; A[n+1,:] .= 0; A[n+1,n+1] = 1  # Dirichlet bcs

# homogeneous Dirichlet boundary conditions
f = pi^2*sin.(pi*x); u_exact = sin.(pi*x)
#f = exp.(x).*(x.^2 .+ 4x .+ 1); u_exact = (1 .- x.^2).* exp.(x)

b = construct_b(n, p, C, M, f, "dirichlet")

u = A \ b  # solve
u = C*[u; zeros(∞)]  # construct solution

#L2 = L2_norm_1D(C \ u - C \ u_exact, M) # is what we want but this computation hangs
L2 = sqrt(L2_norm_1D(C\u_exact, M)^2 + L2_norm_1D(C\u, M)^2 - 2*(C\u)'*M*(C\u_exact))
L2 = round(L2, sigdigits=3) # round to 3 significant digits

# plot
r_ = range(-1, 1; length=1000)
plot(r_, u[r_], legend=false, xlabel=L"x", ylabel=L"u", title="L^2 = $L2")