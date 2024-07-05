#######
# For more information read the paper "Computing and Estimating Kronecker coefficients"
####### 


class AlternantTerm:
    def __init__(self, m, n, perm):
        self.m = m
        self.n = n
        self.perm = perm
        
        
    @staticmethod
    def dominates(v1, v2):
        """
        Checks if a tuple v1 'dominates' another tuple v2. That is,
        checks if each partial sum of v1 is at least the partial sum
        of v2. The partial sums of a tuple are just the sums of the first
        k elements for all k from 1 to the length of v1. 
        
        :param v1: a tuple
        :param v2: a tuple
        :return: a bool
        """
        l = len(v1)
        return all([sum(v1[:i]) >= sum(v2[:i]) for i in range(1, l+1)])

    def __le__(self, alternant_term):
        """
        Returns True if the vector partition function evaluation of self is always at most the 
        vector partition function evaluation of alternant_term regardless of the choice of 
        lmbda, mu, nu.
        We do this by checking domination of the s & t vectors of both.
        
        :param alternant_term: an AlternantTerm
        :return: a bool
        """
        
        bool1 = all([self.dominates(self.s()[k], alternant_term.s()[k]) for k in range(self.m)])
        bool2 = all([self.dominates(self.t()[k], alternant_term.t()[k]) for k in range(1, self.n-1)])
        
        return bool1 and bool2
    
    def __lt__(self, alternant_term):
        """
        Returns True if the vector partition function evaluation of self is always less than the 
        vector partition function evaluation of alternant_term regardless of the choice of 
        lmbda, mu, nu.
        We do this by checking domination of the s & t vectors of both.
        
        :param alternant_term: an AlternantTerm
        :return: a bool
        """
        
        if self <= alternant_term and self != alternant_term:
            return True
        
        else:
            return False
        
    def __ge__(self, alternant_term):
        """
        Returns True if the vector partition function evaluation of self is always at least the 
        vector partition function evaluation of alternant_term regardless of the choice of 
        lmbda, mu, nu.
        We do this by checking domination of the s & t vectors of both.
        
        :param alternant_term: an AlternantTerm
        :return: a bool
        """
        return not self < alternant_term
    
    def __gt__(self, altnerant_term):
        """
        Returns True if the vector partition function evaluation of self is always greater than the 
        vector partition function evaluation of alternant_term regardless of the choice of 
        lmbda, mu, nu.
        We do this by checking domination of the s & t vectors of both.
        
        :param alternant_term: an AlternantTerm
        :return: a bool
        """
        return not self <= alternant_term

    def b_value_dominates(self, alternant_term, lmbda, mu, nu):
        """
        Checks if the b value associated to self and the partitions lmbda, mu, nu is at least as 
        large as the b value as associated to alternant_term and the partitions lmbda, mu, nu
        in each coordinate.
        
        :param alternant_term: an AlternantTerm
        :param lmbda: a partition of length m*n
        :param mu: a partition of length m
        :param nu: a partition of length n
        :return: a bool
        """
        b1 = self.vpf_input(lmbda, mu, nu)
        b2 = alternant_term.vpf_input(lmbda, mu, nu)

        if all([x > y for (x,y) in zip(b1, b2)]):
            return True

        else:
            return False

    def delta(self):
        """
        Returns the partition delta_mn = (mn-1, mn-2, ..., 1, 0).
        
        :return: a list of length m*n
        """
        m,n = self.m, self.n
        return tuple([m*n - i - 1 for i in range(m*n)])

    def part_sum(self, part_vector, lmbda):
        """
        Returns a linear combination of the parts of (lambda + delta_mn).
        
        :param part_vector: an iterable of length m*n
        :param lmbda: a partition of length m*n
        :return: a non-negative integer
        """
        delta = self.delta()

        ps = 0
        for i, coeff in enumerate(part_vector):
            ps += coeff * (lmbda[i] + delta[i])
        return ps

    def indices_to_parts(self, indices):
        """
        Converts a list of indices into a binary vector
        whose k-th index is 1 if k+1 is in the indices list
        (the discrepency is due to Pythonic notation).

        :param indices: a list of non-negative integers
        :return: a binary vector of length m*n
        """
        m,n = self.m, self.n
        return vector([i+1 in indices for i in range(m*n)])

    def x_indices(self):
        """
        Computes the parts of lambda coming from the exponents of x for the alternant term 'self' 
        from the LHS of the Jacobi-Trudi identity.

        :return: a dictionary, key=positive integer in 1,..,m-1, val=list of non-negative integers
        """
        m, n, perm = self.m, self.n, self.perm
        x_inds = {}
        for i in range(1, m):
            x_inds[i] = [perm[i]] + perm[m + (i) * (n - 1):m + (i + 1) * (n - 1)]
        return x_inds

    def y_indices(self):
        """
        Computes the parts of lambda coming from the exponents of y for the alternant term 'self' 
        from the LHS of the Jacobi-Trudi identity.

        :return: a dictionary, key=positive integer in 1,..,n-1, val=list of non-negative integers
        """
        m, n, perm = self.m, self.n, self.perm
        y_indices = {}
        for j in range(1, n):
            y_indices[j] = [perm[m-1+j]]

            for k in range(1, m):
                y_indices[j].append(perm[m-1+j+k*(n-1)])

        return y_indices

    def x(self):
        """
        Represents the output of x_indices using a binary vector.

        :return: a dictionary where key is an integer from 0 to m-1 & val is binary vector of length m-1 (?)
        """
        x_parts = {}
        for i in range(1, self.m):
            x_parts[i] = self.indices_to_parts(self.x_indices()[i])
        return x_parts

    def y(self):
        """
        Represents the output of y_indices using a binary vector.
        
        :return: a dictionary where key is an integer from 0 to n-1 & val is binary vector of length m*n.
        """
        y_parts = {}
        for i in range(1, self.n):
            y_parts[i] = self.indices_to_parts(self.y_indices()[i])
        return y_parts

    def s(self):
        """
        Computes the parts of lambda from the vector l_s. note that no constants appear here since they
        can be computed if we know the parts of lambda - each lambda_i comes with m*n - i.
        For example the output
        {0: (0, 0, 1, 1), 1: (0, 1, 1, 2)}
        corresponds to the 
        l_s(lambda)  = (lambda_3 + lambda_4 + 1, lambda_2 + lambda_3 + 2lambda_4 + 3)

        :return: a dictionary where key is an integer from 0 to m-1 & val is a vector of length m*n with non-negative integer coordinates.
        """
        m,n = self.m, self.n
        s_parts = {}
        s_parts[0] = sum(self.y()[j] for j in range(1, n))

        for k in range(1, m):
            s_parts[k] = sum(self.x()[i] for i in range(k,m)) + sum(self.y()[j] for j in range(1, n))

        return s_parts

    def t(self):
        """
        Computes the parts of lambda from the vector l_t. note that no constants appear here since they
        can be computed if we know the parts of lambda - each lambda_i comes with m*n - i.

        :return: a dictionary where key is an integer from 1 to n-2 & val is a vector of length m*n with non-negative integer coordinates.
        """
        m, n = self.m, self.n
        t_parts = {}
        for l in range(1, n - 1):
            t_parts[l] = (sum(i * self.x()[i] for i in range(1, m)) +
                          sum(self.y()[j] for j in range(l + 1, n)) +
                          (m - 1) * sum(self.y()[j] for j in range(1, n)))

        return t_parts

    def sign(self):
        """
        Returns the sign associated to 'self'. If this sign is positive, the alternant term
        will always make a non-negative contribution to the Kronecker coefficient. If it is negative
        the contribution will always be non-positive.
        
        :return: +/- 1
        """
        return self.perm.sign()  

    def vpf_input(self, lmbda, mu, nu):
        """
        Returns the vpf input associated to alternant term 'self' and the partitions lmbda, mu, nu. 
        This is the value which will be evaluated by the vector partition function in order to 
        determine the contribution of alternant term 'self' to the Kronecker coefficient.
        
        :return: a list of length m+n-2.
        """
        return [r-l for r,l in zip(self.rhs_powers(mu, nu), self.lhs_powers(lmbda))]
    
    def symbolic_vpf_input(self):
        """
        Computes the vpf input for the alternant term 'self' as a function of the parts
        of lmbda, mu, nu. 
        For example, in the K22 case, for at = K22.alternant_terms[0] (the atomic KC)
        the output is
        [-l3 - l4 + n2, -l2 - l3 - 2*l4 + m2 + n2]
        
        
        :return: a list of length m+n-2 whose entries are degree 1 functions in the parts of
        lmbda, mu, nu.
        """
        m,n = self.m, self.n
        
        lmbda = var(' '.join([f'l{i}' for i in range(1, m*n+1)]))
        mu = var(' '.join([f'm{i}' for i in range(1, m+1)]))
        nu = var(' '.join([f'n{i}' for i in range(1, n+1)]))
        return self.vpf_input(lmbda, mu, nu)
    
    def alpha_beta(self):
        """
        Computes (alpha, beta).
        
        :return: a list of length m+n-2
        """
        n,m = self.n, self.m
        
        the_constants = []
        
        # 1st coordinate comes from s0
        alpha_0 = 1/2*(m*n + n - m - 2)*(n-1)*(m-1)
        the_constants.append(alpha_0)
        
        # coordinates 1,...,m-1 come from sa
        for u in range(1,m):
            alpha_u = 1/2*(u^2*n - 2*u*n*m + 2*n*m^2 - u^2 + u - n - 2*m + 2)*(n-1)
            the_constants.append(alpha_u)
            
        # coordinates m+1,..,n+m-2 come from tb
        for v in range(1,n-1):
            beta_v = 1/12*(8*n^2*m^2 - 6*v*n*m + 5*n^2*m - 10*n*m^2 + 6*v^2 - 12*v*n + 6*v*m - 19*n*m + 2*m^2 + 18*v + 14*m)*(m-1)
            the_constants.append(beta_v)
            
        return the_constants

    def lhs_powers(self, lmbda):
        """
        Returns a list which is the evaluation (l_s(lmbda; sigma), l_t(lmbda; sigma))
        where 'sigma' is the permutation of self (i.e self.perm).
        
        :return: a list of non-negative integers
        """
        m,n = self.m, self.n
        powers = []
        for i in range(m):
            powers.append(self.part_sum(self.s()[i], lmbda))

        for j in range(1, n-1):
            powers.append(self.part_sum(self.t()[j], lmbda))

        return powers
    
    l_st = lhs_powers  # this is the name we use in the paper.
    
    def rhs_powers(self, mu, nu, no_constants = False):
        """
        Returns a list which is the evaluation (r_s(mu, nu), r_t(mu, nu)) + (alpha, beta)
        where 'sigma' is the permutation of self (i.e self.perm).
        
        :return: a list of non-negative integers
        """
        return [x+y for x,y in zip(self.r_st(mu, nu), self.alpha_beta())]
            
    def r_st(self, mu, nu):
        """
        Computes (r_s(mu, nu), r_t(mu, nu)). 
        
        :param mu: a partition of length at most m
        :param nu: a partition of length at most n
        
        :return: a list of length m+n-2
        """
        m, n = self.m, self.n
        
        # 1st coordinate comes from s0
        mu_part = 0
        nu_part = sum(nu) - nu[0]
        
        
        constant_part = binomial(n - 1, 2)
        r_st_coords = [mu_part + nu_part + constant_part]
        
        # coordinates 1 <= u <= m  comes from s1 to sn
        for u in range(1, m):
            mu_part = sum([mu[i] for i in range(u, m)])
            nu_part = sum(nu) - nu[0]
            
            constant_part = binomial(m - u, 2) + binomial(n - 1, 2)
            r_st_coords.append(mu_part + nu_part + constant_part)
         
        # coordinates m+1 to m+n-2  (i.e. m+v for 1 <= v <= n-2) comes from t1 to tm-2
        for v in range(1, n-1):
            mu_part = sum([i*mu[i] for i in range(1, m)])
            nu_part = (m-1)*sum([nu[j] for j in range(1, n)]) + sum([nu[j] for j in range(v+1, n)])
            
            constant_part = binomial(m, 3) + (m-1)*binomial(n-1, 2) + binomial(n-v-1, 2)
            r_st_coords.append(mu_part + nu_part + constant_part)
            
        return r_st_coords
    
    def lmn_coeff_vectors(self):
        m, n = self.m, self.n
        
        coeff_vectors = []
        
        lmbda = var(' '.join([f'l{i}' for i in range(1, m*n+1)]))
        mu = var(' '.join([f'm{i}' for i in range(1, m+1)]))
        nu = var(' '.join([f'n{i}' for i in range(1, n+1)]))
        
        for expr in self.symbolic_vpf_input():
            lmbda_coeffs = [expr.coefficient(lmbda[i]) for i in range(m*n)]
            mu_coeffs = [expr.coefficient(mu[i]) for i in range(m)]
            nu_coeffs = [expr.coefficient(nu[i]) for i in range(n)]
            
            coeff_vectors.append(tuple(lmbda_coeffs + mu_coeffs + nu_coeffs))
            
        return coeff_vectors
        