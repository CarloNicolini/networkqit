import sympy as sp


def do_replacement(var, direction):
    xi = sp.Symbol("x_i", positive=True, real=True)
    xj = sp.Symbol("x_j", positive=True, real=True)
    xij = sp.Symbol("x_{ij}", positive=True, real=True)

    yi = sp.Symbol("y_i", positive=True, real=True)
    yj = sp.Symbol("y_j", positive=True, real=True)
    yij = sp.Symbol("y_{ij}", positive=True, real=True)

    if direction is "forward":
        return (
            var.replace(betai, -sp.log(yi))
            .replace(betaj, -sp.log(yj))
            .replace(alphai, -sp.log(xi))
            .replace(alphaj, -sp.log(xj))
            .replace(alphaij, -sp.log(xij))
            .replace(betaij, -sp.log(yij))
        )
    elif direction is "backward":
        return (
            var.replace(yi, sp.exp(-betai))
            .replace(yj, sp.exp(-betaj))
            .replace(xi, sp.exp(-alphai))
            .replace(xj, sp.exp(-alphaj))
            .replace(xij, sp.exp(-alphaij))
            .replace(yij, sp.exp(-betaij))
        )


if __name__ == "__main__":
    sp.init_printing()
    # Define symbols
    gij = sp.Symbol("g_{ij}", positive=True, real=True)
    aij = sp.Symbol("a_{ij}", positive=True, real=True)
    L = sp.Symbol("L", positive=True, real=True)
    wij = sp.Symbol("w_{ij}", positive=True, real=True)
    W = sp.Symbol("W", positive=True, real=True)
    t = sp.Symbol("t", positive=True, real=True)

    alpha = sp.Symbol("\\alpha", positive=True, real=True)
    alphai = sp.Symbol("\\alpha_i", positive=True, real=True)
    alphaj = sp.Symbol("\\alpha_j", positive=True, real=True)
    alphaij = sp.Symbol("\\alpha_{ij}", real=True)

    beta = sp.Symbol("\\beta", positive=True, real=True)
    betai = sp.Symbol("\\beta_i", positive=True, real=True)
    betaj = sp.Symbol("\\beta_j", positive=True, real=True)
    betaij = sp.Symbol("\\beta_{ij}", real=True)

    case = "ctERG"
    # discrete models
    if case is "ER":  # erdos-renyi
        H = alpha * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, 1))
    elif case is "CM":  # configuration model
        H = (alphai + alphaj) * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, 1))
    elif case is "WRG":  # weighted random graph
        H = beta * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, sp.oo))
    elif case is "WCM":  # weighted configuration model
        H = (betai + betaj) * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, sp.oo))
    elif case is "ERG":
        H = alpha * sp.Heaviside(gij) + beta * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, sp.oo))
    elif case is "ECM":
        H = (alphai + alphaj) * sp.Heaviside(gij) + (betai + betaj) * gij
        Z = sp.summation(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, sp.oo))

    # weighted links
    if case in "cWRG":
        H = beta * gij
    elif case is "cWCM":
        H = (betai + betaj) * gij
    elif case is "ctERG":
        H = (alpha) * t * sp.Heaviside(gij - t) + beta * gij * sp.Heaviside(gij - t)
    elif case is "ctWRG":
        H = beta * sp.Heaviside(gij - t)
    elif case is "ctWCM":
        H = (betai + betaj) * sp.Heaviside(gij - t)
    elif case is "ctECM":
        H = (alphai + alphaj) * t * sp.Heaviside(gij - t) + (
            betai + betaj
        ) * sp.Heaviside(gij - t)

    gmax = sp.Symbol("g_{max}", real=True, positive=True)
    Z = sp.integrate(sp.exp(-H.rewrite(sp.Piecewise)), (gij, 0, gmax))

    # Probability
    P = sp.exp(-H) / Z

    # Loglikelihood
    logL = sp.expand_log(sp.log(P))
    print(sp.simplify(logL.rewrite(sp.Piecewise)))
    # Free energy
    F = -sp.log(Z)
    # print(sp.limit(F,gmax,+sp.oo))
    # Expectation
    # print(sp.simplify(sp.diff(F,beta)))
    # print(sp.simplify(do_replacement(sp.diff(F,betai),'forward')))
