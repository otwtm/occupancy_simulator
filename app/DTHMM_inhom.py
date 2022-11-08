import numpy as np

import scipy.linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class DTMC_inhom:
    def __init__(self, dim, y_dim, n_cycle, spline_basis, hidden=True):
        self.y_dim = y_dim
        self.dim = dim
        self.n_cycle = n_cycle
        self.spline_basis = spline_basis
        self.hidden = hidden
        #self.Gamma = np.zeros((n_cycle, dim, dim))
        #self.param = np.ones((dim, dim-1, n_cycle))
        self.delta = np.zeros(dim)
        self.B = None # SDD, B[i,j] is P(Y_t = y_j | X_t = i)
        #self.LL = 0
        if not self.hidden:
            self.B = np.eye(self.dim)

    def calc_Gamma(self, param):
        Gamma = np.zeros((self.n_cycle, self.dim, self.dim))
        for t in range(self.n_cycle):
            Gamma[t, :, :] = np.eye(self.dim)
            for i in range(self.dim):
                for j in range(i):
                    Gamma[t, i, j] = np.exp(param[((self.dim-1)*i+j)*4:((self.dim-1)*i+j+1)*4] @ self.spline_basis[t, :])
                for j in range(i + 1, self.dim):
                    #print("param", param)
                    #print("np.exp(param[i, j-1, :]", np.exp(param[i, j-1, :]))
                    #print("spline_basis[t, :]", self.spline_basis[t, :])
                    Gamma[t, i, j] = np.exp(param[((self.dim-1)*i+j-1)*4:((self.dim-1)*i+j)*4] @ self.spline_basis[t, :])
                Gamma[t, i, :] = Gamma[t, i, :] / sum(Gamma[t, i, :])
        self.Gamma = Gamma
        return Gamma


    def set_observations(self, Y):
        self.Y = Y
        self.T_full = len(Y)
        self.Y_binary = np.zeros((self.T_full, self.y_dim))
        for t in range(self.T_full):
            self.Y_binary[t, Y[t]] = 1


    def __str__(self):
        string = "Number of states: {0}\nTransition probability matrix:\n {1}\n". format(self.dim, self.Gamma)
        return string

    def calc_alpha_beta(self):
        """
        forward probabilities
        alpha[t, i]
        """
        c = np.zeros(self.T_full)
        #eta = np.zeros(self.T_full)
        #print("self.T_full", self.T_full)
        #print("self.dim", self.dim)
        alpha = np.zeros((self.T_full, self.dim))
        #print("self.B[:, 0]", self.B[:, 0])
        #print("self.delta", self.delta)
        alpha[0, :] = self.delta * self.B[:, 0]
        #print("alpha[0, :]", alpha[0, :])
        c[0] = 1.0 / sum(alpha[0, :])
        #eta[0] = c[0]
        alpha[0, :] = c[0] * alpha[0, :]
        #print("sum(alpha[0, :])", sum(alpha[0, :]))
        for t in range(self.T_full-1):
            #print("self.Gamma[t%self.n_cycle, :, :]", self.Gamma[t%self.n_cycle, :, :])
            #print("self.B[:, self.Y[t+1]]", self.B[:, self.Y[t+1]])
            #print("alpha[t, :]", alpha[t, :])
            #print("alpha[t, :] @ self.Gamma[t%self.n_cycle, :, :]", alpha[t, :] @ self.Gamma[t%self.n_cycle, :, :])
            alpha[t+1, :] = self.B[:, self.Y[t+1]] * (alpha[t, :] @ self.Gamma[(t+1)%self.n_cycle, :, :])
            #print("alpha[t+1, :]", alpha[t+1, :])
            c[t+1] = 1.0/sum(alpha[t+1, :])
            #print("c[t]: ", c[t])
            #eta[t+1] = eta[t] * c[t+1]
            alpha[t+1, :] = c[t+1]*alpha[t+1, :]
            #print("sum over scaled alphas", sum(alpha[t+1, :]))
        #print("alpha", alpha)
        beta = np.zeros((self.T_full, self.dim))
        beta[self.T_full-1, :] = np.ones(self.dim)
        beta[self.T_full-1, :] = c[self.T_full-1] * beta[self.T_full-1, :]
        for t in range(self.T_full-2, -1, -1):
            for i in range(self.dim):
                XX = beta[t + 1, :]
                XX = XX.astype(float)
                YY = (self.Gamma[(t+1)%self.n_cycle, i, :].transpose() * self.B[:, self.Y[t + 1]])
                YY = YY.astype(float)
                beta[t, i] = XX @ YY
            beta[t, :] = beta[t, :] * c[t]
        #print("beta", beta)
        return alpha, beta, c


    def calc_uhat(self, alpha, beta):
        uhat = np.zeros((self.T_full, self.dim))
        for t in range(self.T_full):
            denomi = alpha[t, :] @ beta[t, :]
            for i in range(self.dim):
                uhat[t, i] = (alpha[t, i]*beta[t, i])/denomi
        return uhat

    def calc_vhat(self, alpha, beta, c):
        vhat = np.zeros((self.T_full-1, self.dim, self.dim))
        for t in range(self.T_full-1):
            denomi = (alpha[t, :] @ beta[t, :]) /c[t]
            for i in range(self.dim):
                for j in range(self.dim):
                    vhat[t, i, j] = (alpha[t, i]*self.Gamma[t%self.n_cycle, i, j]*beta[t+1, j]*self.B[j, self.Y[t+1]])/denomi
        return vhat

    def EStep(self, uhat, vhat):
        """
        Gamma = np.zeros((self.n_cycle, self.dim, self.dim))
        for t in range(self.n_cycle):
            for i in range(self.dim):
                for j in range(self.dim):
                    #print("sum(vhat[:self.T_full-1, i, j])", sum(vhat[:self.T_full-1, i, j]))
                    #print("sum(uhat[:self.T_full-1, i])", sum(uhat[:self.T_full-1, i]))
                    Gamma[t, i, j] = sum(vhat[:self.T_full-1, i, j])/(sum(uhat[:self.T_full-1, i]))

        """

        def objective_fun(param, vhat):
            Gamma = self.calc_Gamma(param)
            sum = 0
            for t in range(self.T_full-1):
                for i in range(self.dim):
                    for j in range(self.dim):
                        #print("t", t)
                        #print("t%self.n_cycle", t%self.n_cycle)
                        sum = sum + vhat[t, i, j] * np.log(Gamma[t%self.n_cycle, i, j])
            return -sum

        Nfeval = 1
        def callbackF(xk):
            nonlocal Nfeval
            print("iteration: {}".format(Nfeval))
            print("likelihood: {}".format(objective_fun(param=xk, vhat=vhat)))
            Nfeval += 1
        method = 'BFGS'
        if method == "Nelder-Mead":
            options = {'xatol': 0.1, 'fatol': 1.0, 'disp': True, 'maxiter': 800}
        elif method == "BFGS":
            options = {'gtol': 0.5, 'disp': True, 'maxiter': 30}
        res = minimize(objective_fun, x0=self.param, args=vhat, method=method, callback=None, options=options, tol=None)
        self.param = res.x
        #print("res.x", res.x)
        Gamma = self.calc_Gamma(self.param)

        delta = self.Y_binary[0, :]
        B = self.B
        return [delta, Gamma, B]

    def MStep(self):
        alpha, beta, c = self.calc_alpha_beta()
        uhat = self.calc_uhat(alpha, beta)
        vhat = self.calc_vhat(alpha, beta, c)
        self.calc_log_lik(c)
        return [uhat, vhat]

    def calc_log_lik(self, c):
        self.LL = -np.sum(np.log(c))
        return self.LL

    def baum_welch(self, param0, delta0=None, B0=None, max_iter=30, tol=0.1):
        #if self.hidden == True and self.B == None:
        #    print("Please provide the argument B0")
        self.param = param0
        self.Gamma = self.calc_Gamma(self.param)
        if self.hidden:
            self.delta = delta0
            self.B = B0
        iter = 0
        self.LL = 0
        previous_LL = 0
        while True:
            print("iteration: ", iter, "   LL: ", self.LL)
            uhat, vhat = self.MStep()
            dummy, self.Gamma, dummy = self.EStep(uhat, vhat)
            current_LL = self.LL
            if current_LL<previous_LL and iter>1:
                print("Warning: Decreasing Log-Likelihood")
            if iter>max_iter:
                print("Maximum iterations reached.")
                break
            if (abs(current_LL - previous_LL) < tol and iter > 2):
                print("Likelihood reached tolerance threshold.")
                break
            previous_LL = current_LL
            iter=iter+1
        self.Gamma = self.calc_Gamma(self.param)
        #print("\nEstimated Gamma:\n", self.Gamma)
        print("\nEmission Matrix:\n", self.B)


    def simulate_sequence(self, N, initial_state=0):
        states = np.zeros(N).astype(int)
        states[0] = np.random.choice(range(self.dim), p=self.delta)
        observations = np.zeros(N).astype(int)
        observations[0] = np.random.choice(range(self.y_dim), p=self.B[states[0], :])
        for t in range(1,N):
            states[t] = np.random.choice(range(self.dim), p=self.Gamma[t%self.n_cycle, states[t - 1], :])
            observations[t] = np.random.choice(range(self.y_dim), p=self.B[states[t], :])
        return states, observations


    def viterbi(self):
        """
        Notation as in Wikipedia
        :return: most likely state sequence
        """
        self.B = self.B.astype(float)
        T1 = np.zeros((self.dim, self.T_full))
        T2 = np.zeros((self.dim, self.T_full))
        z = np.zeros(self.T_full).astype(int)
        #print("type(self.B)", type(self.B))
        #print("type(self.delta)", type(self.delta))
        #print("np.log(self.delta)", np.log(self.delta))
        #print("self.B", self.B)
        #print("self.Y", self.Y)
        #print("type(self.B[:, self.Y[0]])", type(self.B[:, self.Y[0]]))
        #print("np.log(self.B[:, self.Y[0]]", np.log(self.B[:, self.Y[0]]))
        T1[:, 0] = np.log(self.delta) * np.log(self.B[:, self.Y[0]])
        #print("self.delta * self.B[:, self.Y[0]]", self.delta * self.B[:, self.Y[0]])
        #print("self.delta", self.delta)
        #print("self.Y[0]", self.Y[0])
        T2[:, 0] = np.zeros(self.dim)
        for t in range(1, self.T_full):
            for i in range(self.dim):
                #print("T1[:, t-1]", T1[:, t-1])
                #print("self.Gamma[t%self.n_cycle, :, i]", self.Gamma[t%self.n_cycle, :, i])
                #print("np.log(self.B[i, self.Y[t]])", np.log(self.B[i, self.Y[t]]))
                temp = T1[:, t-1] + np.log(self.Gamma[t%self.n_cycle, :, i]) + np.log(self.B[i, self.Y[t]])
                #print("temp", temp)
                #print("temp", temp)
                T1[i, t] = np.max(temp)
                T2[i, t] = np.argmax(temp)
            #print("T1[:, t]", T1[:, t])
            #print("T2[:, t]", T2[:, t])
        z[self.T_full-1] = np.argmax(T1[:, self.T_full-1])
        #print("T1[:, self.T_full-1]", T1[:, self.T_full-1])
        for t in range(self.T_full-1, 0, -1):
            z[t-1] = T2[z[t], t]
        print("z:", z)
        return z

    '''
    def viterbi(self):
        """
        Notation as in Wikipedia
        :return: most likely state sequence
        """
        T1 = np.zeros((self.dim, self.T_full))
        T2 = np.zeros((self.dim, self.T_full))
        z = np.zeros(self.T_full).astype(int)
        print("self.delta", self.delta)
        print("self.B", self.B)
        print("self.Y", self.Y)
        T1[:, 0] = self.delta * self.B[:, self.Y[0]]
        print("self.delta * self.B[:, self.Y[0]]", self.delta * self.B[:, self.Y[0]])
        print("self.delta", self.delta)
        print("self.Y[0]", self.Y[0])
        T2[:, 0] = np.zeros(self.dim)
        for t in range(1, self.T_full):
            for i in range(self.dim):
                #print("T1[:, t-1]", T1[:, t-1])
                #print("self.Gamma[t%self.n_cycle, :, i]", self.Gamma[t%self.n_cycle, :, i])
                #print("self.B[i, self.Y[t]]", self.B[i, self.Y[t]])
                temp = T1[:, t-1] * self.Gamma[t%self.n_cycle, :, i] * self.B[i, self.Y[t]]
                #print("temp", temp)
                T1[i, t] = np.max(temp)
                T2[i, t] = np.argmax(temp)
            print("T1[:, t]", T1[:, t])
            print("T2[:, t]", T2[:, t])
        z[self.T_full-1] = np.argmax(T1[:, self.T_full-1])
        print("T1[:, self.T_full-1]", T1[:, self.T_full-1])
        for t in range(self.T_full-1, 0, -1):
            z[t-1] = T2[z[t], t]
        print("z:", z)
        return z
    '''

#######################################################################################################################



if __name__ == "__main__":
    import os
    import pandas as pd

    absFilePath = os.path.abspath(__file__)
    print(absFilePath)
    fileDir = os.path.dirname(os.path.abspath(__file__))
    print(fileDir)
    parentDir = os.path.dirname(fileDir)
    print(parentDir)
    grandparentDir = os.path.dirname(parentDir)
    print(grandparentDir)

    spline_basis = pd.read_csv(os.path.join(grandparentDir, 'R/spline_basis.csv'))
    spline_basis = np.array(spline_basis)



    ## true parameters
    Gamma_true = np.zeros((144, 2, 2))
    for t in range(144):
        Gamma_true[t, :, :] = np.array([[0.5, 0.5], [0.3, 0.7]])
    #print("Gamma_true\n", Gamma_true)
    B_true = np.array([[1.0, 0.0],[0.0, 1.0]])
    #B_true = np.array([[1, 0],[0, 1]])
    delta_true = [1.0, 0.0]

    dtmc = DTMC_inhom(dim=2, y_dim=2, n_cycle=144, hidden=True)
    dtmc.Gamma = Gamma_true
    dtmc.B = B_true
    dtmc.delta = delta_true
    seq = dtmc.simulate_sequence(N=2000)
    observations = seq[1]

    ## initial parameters
    Gamma0 = np.zeros((144, 2, 2))
    for t in range(144):
        Gamma0[t, :, :] = np.array([[0.1, 0.9], [0.8, 0.2]])

    m=2
    param0 = np.ones(m*(m-1)*4)
    print("param0", param0)

    B0 = B_true
    delta0 = delta_true

    dtmc.set_observations(Y=seq[0])
    dtmc.baum_welch(param0=param0, B0=B0, delta0=delta0, max_iter=20)
    # dtmc.viterbi()
    
    plt.plot(dtmc.Gamma[:, 1, :])
    plt.show()


    """   
    Gamma_true = np.array([[0.5, 0.3, 0.2], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6]])
    B_true = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    delta_true = [1.0, 0.0, 0.0]

    dtmc = DTMC_inhom(dim=3, y_dim=2, n_cycle=144, hidden=True)
    dtmc.Gamma = Gamma_true
    dtmc.B = B_true
    dtmc.delta = delta_true
    seq = dtmc.simulate_sequence(N=20)
    observations = seq[1]
    print(seq)

    ## initial parameters
    Gamma0 = Gamma_true
    B0 = B_true
    delta0 = delta_true

    dtmc.set_observations(Y=seq[1])
    dtmc.baum_welch(Gamma0=Gamma0, B0=B0, delta0=delta0)
    #dtmc.viterbi()
    """






