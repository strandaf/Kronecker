load('AlternantTerm.sage')
load('PiecewiseQuasipolynomial.sage')

from itertools import combinations
from collections import Counter

class KroneckerComputer:
    def __init__(self, m, n, alternant_terms, vector_partition_function):
        self.m = m
        self.n = n
        self.alternant_terms = alternant_terms
        self.vector_partition_function = vector_partition_function

    def alternant_terms_by_evaluation(self, lmbda, mu, nu):
        """
        Returns a record of the vector partition evaluations associated to each
        AlternantTerm when computing the Kronecker coefficient of lmbda, mu, nu.
        :param lmbda: partition of length at most m*n
        :param mu: partition of length at most m
        :param nu: partition of length at most n
        :return: dictionary, key=integer, val=list of alternant terms.
        """
        d = {}
        vp = self.vector_partition_function
        for alternant_term in self.alternant_terms:
            b = alternant_term.vpf_input(lmbda, mu, nu)
            val = vp.evaluate(b)*alternant_term.sign()
            d.setdefault(val, []).append(alternant_term)
        return d
    
    def atomic_kronecker_coefficient(self, lmbda, mu, nu):
        """
        Returns the atomic Kronecker coefficient for partitions lmbda, mu, nu.
        :param lmbda: partition of length at most m*n
        :param mu: partition of length at most m
        :param nu: partition of length at most n
        """
        atomic_alternant_term = self.alternant_terms[0]
        b = atomic_alternant_term.vpf_input(lmbda, mu, nu)
        vp = self.vector_partition_function
        return vp.evaluate(b)
    
    def bounds(self, lmbda, mu, nu):
        """
        Returns the bounds from Corollary 6.6 of our paper "Estimating and computing Kronecker coefficients".
        
        
        :param lmbda: partition of length at most m*n
        :param mu: partition of length at most m
        :param nu: partition of length at most n
        """
        m,n = self.m, self.n
        
        c1 = (n-1)*(m^2-1) - 1
        c2 = (m-1)*(n-1)^2 - 1
        c3 = (n-1)*binomial(m-1, 2) + (m-1) - 1
        
        def f1(i): 
            return 2*binomial(n-1, 2)*(i-2) - 1
        
        def f2(j):
            return (n-j-1)*(m-1) - 1
        
        at = self.alternant_terms[0]
        b = at.vpf_input(lmbda, mu, nu)
        
        P1 = binomial(b[0] + c1, b[0])*binomial(2*b[1] + c2, b[1])*binomial(b[-1] + c3, b[-1])
        P2 = product([binomial(b[i-1] + f1(i), b[i-1]) for i in range(3,m)])
        P3 = product([binomial(b[m+j-1] + f2(j), b[m+j-1]) for j in range(1, n-2)])
        
        return factorial(m*n)*P1*P2*P3/2
    
    def bounds_simple(self, N):
        """
        Returns the bounds from Corollary 6.8 of our paper "Estimating and computing Kronecker coefficients".
        
        
        :param lmbda: partition of length at most m*n
        :param mu: partition of length at most m
        :param nu: partition of length at most n
        """
        m,n = self.m, self.n
        
        c1 = (m^2-1)*(n-1) - 1
        c2 = (m-1)*(n-1)^2 - 1
        c3 = binomial(m-1, 2)*(n-1) + (m-1) - 1
        
        def f1(i): 
            return 2*binomial(n-1, 2)*(i-2) - 1
        
        def f2(j):
            return (n-j-1)*(m-1) - 1
        
        P1 = binomial(N + c1, N)*binomial(2*N + c2, 2*N)*binomial((2*m - 1)*N + c3, (2*m - 1)*N)
        P2 = product([binomial(2*N + f1(i), 2*N) for i in range(3,m+1)])
        P3 = product([binomial((2*m-1)*N + f2(j), (2*m-1)) for j in range(1, n-2)])
        
        return factorial(m*n)*P1*P2*P3/2

    def kronecker_coefficient(self, lmbda, mu, nu):
        """
        Computes the Kronecker coefficient for lambda, mu, nu

        :param lmbda: a partition of length m*n
        :param mu: a partition of length m
        :param nu: a partition of length n
        :return: non-negative integer
        """
        kc = 0
        vp = self.vector_partition_function

        for alternant_term in self.alternant_terms:
            b = alternant_term.vpf_input(lmbda, mu, nu)
            vp_val = vp.evaluate(b)
            kc += vp_val*alternant_term.sign()

        return kc

    def kronecker_coefficient_table(self, lmbda, mu, nu, non_zero_only = True, include_st = False):
        """
        Returns the Kronecker coefficient and the number of alternant terms with a non-zero contribution, as well as
        a table storing 
        - the permutation for each alternant term
        - the s & t vectors for the alternant term (optional, by default this is not shown)
        - the b value for that alternant term at lmbda, mu, nu
        - the evaluation of the b value by the vector partition function
        One can optionally show only the alternant terms with a non-zero contribution (this is the default)
        or all of the alternant terms.

        :param lmbda: a partition of length m*n
        :param mu: a partition of length m
        :param nu: a partition of length n
        :param non_zero_only: a bool
        :param include_st: a bool
        :return: a non-negative integer, a table, and a positive integer.
        """
        kc = 0
        vp = self.vector_partition_function
        num_terms = 0

        table_rows = []
        for alternant_term in self.alternant_terms:
            b = alternant_term.vpf_input(lmbda, mu, nu)
            vp_val = vp.evaluate(b)
            kc += vp_val*alternant_term.sign()

            if vp_val != 0:
                num_terms += 1
                
            if vp_val != 0 or not non_zero_only:
                
                if include_st:
                    table_rows.append([alternant_term.perm,
                                       alternant_term.s(), alternant_term.t(),
                                       tuple(b), vp_val*alternant_term.sign()])
                    
                else:
                    table_rows.append([alternant_term.perm,
                                       tuple(b), vp_val*alternant_term.sign()])

        table_rows.sort(key=lambda row: -row[-1]^2)
        
        if include_st:
            header = [['Permutation', 's', 't', 'b value', 'Signed evaluation']]
            
        else:
            header = [['Permutation', 'b value', 'Signed evaluation']]
            
        table_rows = header + table_rows

        return kc, num_terms, table(table_rows, header_row=True)
    
    def kronecker_coefficient_chamber_table(self, lmbda, mu, nu):
        """
        Returns a table storing 
        - the permutation for each alternant term
        - the b value for that alternant term at lmbda, mu, nu
        - a list of the chambers in which this b value takes place
        - the evaluation of the b value by the vector partition function

        :param lmbda: a partition of length m*n
        :param mu: a partition of length m
        :param nu: a partition of length n
        :return: a non-negative integer, a table, and a positive integer.
        """
        kc = 0
        vp = self.vector_partition_function
        num_terms = 0

        table_rows = []
        for alternant_term in self.alternant_terms:
            b = alternant_term.vpf_input(lmbda, mu, nu)
            vp_val, chambers = vp.evaluate_with_chambers(b)
            kc += vp_val*alternant_term.sign()

            if vp_val != 0:
                num_terms += 1

                table_rows.append([alternant_term.perm, tuple(b), chambers, vp_val*alternant_term.sign()])

        table_rows.sort(key=lambda row: -row[-1]^2)
        header = [['Permutation', 'b value', 'chambers', 'Signed evaluation']]
        table_rows = header + table_rows

        return kc, num_terms, table(table_rows, header_row=True)
    
    def murnaghan_inequality(self):
        """
        Computes Murnaghan's vanishing condition.
        
        :return: a list containing a single tuple.
        """
        m,n = self.m, self.n
        
        ineq = [0 for i in range(m*n + m + n)]
        
        for i in range(1, m*n):
            ineq[i] = -1
            
        for j in range(m*n + 1, m*n + m):
            ineq[j] = 1
            
        for k in range(m*n + m + 1, m*n + m + n):
            ineq[k] = 1
            
        return [tuple([0] + ineq)]
        
    def num_terms(self):
        """
        Computes the number of alternant terms associated to self. Note for K22, K23, K24 we have filtered out
        the alternant terms that never have a non-zero contribution to the Kronecker coefficient.
        
        :return: a positive integer.
        """
        return len(self.alternant_terms)
    
    @staticmethod
    def pad_partition(partition, k):
        """
        Takes a partition of length at most k, and returns the version of
        the partition padded with enough zeroes to make the length k.
        
        :param partition: a partition of length at most k 
        :param k: a positive integer
        :return: a 'padded' partition of length k
        """
        q = k - len(partition)
        return partition + [0 for i in range(q)]
    
    def partition_equalities(self):
        """
        Returns the tuples associated to the equalities
        |lambda| = |mu| = |nu|
        """
        m,n = self.m,self.n
        
        partition_equalities = []
        starter = [0 for i in range(m*n + m + n)]
        
        # |lambda| = |mu|
        eq = [x for x in starter]
        for i in range(m*n+m):
            eq[i] = (-1)^int(i < m*n)
        partition_equalities.append(eq)
        
        eq = [x for x in starter]
        for i in range(m*n, m*n+m+n):
            eq[i] = (-1)^int(i < m*n + m)
        partition_equalities.append(eq)
            
        return [tuple([0] + pq) for pq in partition_equalities]
        
    def partition_inequalities(self):
        """
        [FILL THIS IN!!!!!!]
        """
        m,n = self.m,self.n
        
        partition_inequalities = []
        starter = [0 for i in range(m*n + m + n)]  # a list from which we will generate all the tuples
        
        # lambda inequalities
        for i in range(m*n-1):
            ineq = [x for x in starter]
            ineq[i] = 1
            ineq[i+1] = -1
            partition_inequalities.append(ineq)
        
        ineq = [x for x in starter]
        ineq[m*n-1] = 1
        partition_inequalities.append(ineq)
        
        # mu inequalities
        for i in range(m*n, m*n + m - 1):
            ineq = [x for x in starter]
            ineq[i] = 1
            ineq[i+1] = -1
            partition_inequalities.append(ineq)
            
        ineq = [x for x in starter]
        ineq[m*n+m-1] = 1
        partition_inequalities.append(ineq)
        
        # nu inequalities
        for i in range(m*n+m, m*n + m + n - 1):
            ineq = [x for x in starter]
            ineq[i] = 1
            ineq[i+1] = -1
            partition_inequalities.append(ineq)
            
        ineq = [x for x in starter]
        ineq[-1] = 1
        partition_inequalities.append(ineq)
        
        # need to put a zero in front of each for the RHS of the inequality.
        partition_inequalities1 = [tuple([0] + pi) for pi in partition_inequalities]  
        
        return partition_inequalities1
    
    def pak_panova_bounds_simple(self, N):
        m,n = self.m, self.n
        
        c = (n*m)^2
        
        return (1 + c/N)^N*(1 + N/c)^c
        
    def pak_panova_bounds(self, mu, nu, lmbda):
        a,b,c = len(mu), len(nu), len(lmbda)
        
        mu_bound = product([binomial(mu[i] - (i+1) + b*c, mu[i]) for i in range(a)])
        nu_bound = product([binomial(nu[i] - (i+1) + a*c, nu[i]) for i in range(b)])
        lambda_bound = product([binomial(lmbda[i] - (i+1) + a*b, lmbda[i]) for i in range(c)])
        
        return min(mu_bound, nu_bound, lambda_bound)
    
    def poset(self):
        """
        Returns a poset of the alternant terms where at1 > at2 iff each
        of the s & t vectors of at1 are > the s & t vectors of at2.
        
        :return: a Poset.
        """
        l = len(self.alternant_terms)
        return Poset((list([0..l-1]), lambda i,j : self.alternant_terms[i] <= self.alternant_terms[j]))
    
    def stable_face(self):
        """
        Returns the stable face from Theorem 5.4 (defined by the inequalities 40--42 of Proposition 5.2).
        
        :return: a Cone.
        """
        atomic_at = self.alternant_terms[0]
        eq_vectors = []
        
        for coeff_vector in atomic_at.lmn_coeff_vectors():
            eq_vector = tuple([0] + list(coeff_vector))
            eq_vectors.append(eq_vector)
            
        P = Polyhedron(eqns=eq_vectors + self.partition_equalities(), ieqs=self.partition_inequalities(), base_ring=QQ)
            
        return Cone(P)
    
    def vanishing_conditions(self):
        """
        Returns the set of (non)-vanishing conditions from Theorem 4.4.
        
        :return: list of inequalities.
        """
        atomic_at = self.alternant_terms[0]
        return [c >= 0 for c in atomic_at.symbolic_vpf_input()]
    
    def vanishing_conditions_cone(self):
        m,n = self.m, self.n
        
        atomic_at = self.alternant_terms[0]
        ineq_vectors = []
        
        for coeff_vector in atomic_at.lmn_coeff_vectors():
            ineq_vector = tuple([0] + list(coeff_vector))
            ineq_vectors.append(ineq_vector)
            
        P = Polyhedron(ieqs=ineq_vectors+self.partition_inequalities(), eqns = self.partition_equalities(), base_ring=QQ)
        
        return Cone(P)