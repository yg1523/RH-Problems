using LinearAlgebra
using ClassicalOrthogonalPolynomials, SingularIntegrals, QuadGK
using ContinuumArrays
P = Legendre()

n = 5
#normalize to use Cauchy Operator
stieltjes(P, 0.1+0im)[1:n]/(-2π*im)

x = ClassicalOrthogonalPolynomials.grid(P, n)
C⁺ = stieltjes(P, x .+ 0im)[:,1:n]/(-2π*im) 
C⁻ = stieltjes(P, x .- 0im)[:,1:n]/(-2π*im)

using Test
# evaluate the stiletjes transform of exp(x)
c = transform(P, exp) # Legendre coefficients
f = P*c # Legendre expansion
@test f[0.1] ≈ exp(0.1)

z = 0.1+0.2im
@test stieltjes(f, z) ≈ quadgk(x -> -exp(x)/(x-z), -1, 1)[1]

g = x -> -exp(x)/(x-z)
@time quadgk(g, -1, 1)[1]
@time stieltjes(f, z)







#Application of ContinuumArrays
basis = Legendre()
axes(basis, 1)

x = range(-1, 1, 100)
Q = basis[x, 1:5] #construct a quasimatrix
f = x -> x^2
c = Q\f.(x) #coefficient of approximation

f_pred = Q * c  # 逼近值
f_true = f.(x)

using Plots
plot(x, f_true, label="True f(x)")
plot!(x, f_pred, label="Approximation")


F = ClassicalOrthogonalPolynomials.Fourier()
f2(x) = sin(x) + 0.5cos(2x)
coeffs = expand(F, f2)

N = 20
truncated_coeffs = coeffs[1:N]
f_approx = F[:, 1:N] * truncated_coeffs
f_approx[1.5]