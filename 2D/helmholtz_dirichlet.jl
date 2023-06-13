using PiecewiseOrthogonalPolynomials, Plots, LinearAlgebra, ClassicalOrthogonalPolynomials, StaticArrays, SparseArrays
import PiecewiseOrthogonalPolynomials: ∞

"""
Solving k²<u,v> + <∇u, ∇v> = <f,v> on [-1,1]^2 using standard backslash
"""

include("2D.jl")
include("../1D/1D.jl")

k = 1
n = 4  # amount of cells in each direction, currently only works for 1 cell
dx = 2/n  # width of cell
p = 5  # degree, up to W_p
N = n+1 + n*(p+1)  # amount of basis functions in C

r = range(-1, 1; length=n+1)
C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2) * P_k^(1,1)
P = ContinuousPolynomial{0}(r)  # piecewise Legendre
CtP = C'*P
PtP = P'*P
PinvC = P\C
x = axes(C, 1)
y = axes(C, 1)
D = Derivative(x)
M = C'*C  # mass matrix
D₂ = (D*C)'*(D*C)  # weak Laplacian

M¹ = sparse(M[Block.(1:p+2),Block.(1:p+2)])  # up to polynomial degree p
M¹[1,:] .= 0; M¹[n+1,:] .= 0; M¹[:, 1] .= 0; M¹[:, n+1] .= 0; M¹[1,1] = 1; M¹[n+1, n+1] = 1;
D₂¹ = sparse(D₂[Block.(1:p+2),Block.(1:p+2)])
D₂¹[1,:] .= 0; D₂¹[n+1,:] .= 0; D₂¹[:, 1] .= 0; D₂¹[:, n+1] .= 0; D₂¹[1,1] = 1; D₂¹[n+1, n+1] = 1;

A = k^2*kron(M¹, M¹) + kron(D₂¹, M¹) + kron(M¹, D₂¹)

# finding RHS
u_exact = z -> ((x,y)= z; (1 .- y.^2) .* (1 .- x.^2) .* exp.(x))
f = z -> ((x,y)= z; k^2 .* (1 .- y.^2) .* (1 .- x.^2) .* exp.(x) .- exp.(x) .* (-3 .+ y.^2 .+ 4 .* x .* (-1 .+ y.^2) .+ x.^2 .* (1 .+ y.^2)))
F = interpolate_2D(f, n, p+2)  # interpolate F into P⊗P
b = CtP[1:N, 1:n*p]*F*(CtP[1:N, 1:n*p])'  # RHS <f,v>
b[1,:] .= 0; b[n+1,:] .= 0; b[:, 1] .= 0; b[:, n+1] .= 0  # Dirichlet bcs
b = vec(b)  # vectorise b

U = A \ b  # solve
U = reshape(U, N, N)
u = (x,y) -> C[x, 1:N]'*U*C[y, 1:N]  # construct solution

# computing L2 norm
U_legendre = PinvC[Block.(1:p+2), Block.(1:p+2)]*U*PinvC[Block.(1:p+2), Block.(1:p+2)]' # transform coefficients of numerical solution from C⊗C to P⊗P
U_exact = interpolate_2D(u_exact, n, 2*p+2) # double the interpolation degree of exact solution for accuracy
U_exact[1:size(U_legendre)[1], 1:size(U_legendre)[2]] -= U_legendre  # compute difference
L2 = L2_norm_2D(U_exact, (P'P)[Block.(1:2p+2), Block.(1:2p+2)]) # compute L2 norm of difference
L2 = round(L2, sigdigits=3) # round to 3 significant digits

# plot
x = y = range(-1,1; length=50)
contourf(x, y, u.(x', y), xlabel=L"x", ylabel=L"y", title="L^2 = $L2")