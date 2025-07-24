using ClassicalOrthogonalPolynomials
using SingularIntegrals
using ContinuumArrays
using LinearAlgebra

n = 16
P = Legendre()
x = ClassicalOrthogonalPolynomials.grid(P, n)
G = x -> exp(-40 * x^2)

C⁻ = stieltjes(P, x .- 0im)[:,1:n]/(-2π*im)

A = zeros(ComplexF64, n, n)
f = zeros(ComplexF64, n)

for j in 1:n
    G_j = G(x[j])  # 跳跃条件在配点s_j的值
    for k in 1:n
        # 矩阵项: P_k(s_j) - (G_j - 1) * (C^- P_k)(s_j)
        A[j, k] = P[x[j], k] - (G_j - 1) * C⁻[j, k]
    end
    f[j] = G_j - 1  # 右端项
end

c = A \ f
u_n(s) = sum(c[k] * P[s, k] for k in 1:n)

#verify that this is a good practice of collocation method
residual = [u_n(x[j]) - (G(x[j]) - 1 + (C⁻ * c)[j] * (G(x[j]) - 1)) for j in 1:n]
println("maximum residual: ", maximum(abs.(residual)))
