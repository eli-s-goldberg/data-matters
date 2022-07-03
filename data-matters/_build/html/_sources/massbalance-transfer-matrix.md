# Stock and flow modeling approaches: Eigen town, the special cases of the Markov transfer matrix, and ODEs 'oh my'.

I'm back with more non-machine learning, differential equation based modeling approaches. Why? Well, because I truly
believe that systems analysis modeling approaches are in general the most powerful, most elegant ways to describe our
world.

In reference to this mathematical opinion article, I'd like to remind folks of a phrase my older brother likes to say, "
...I'm the wrong kind of doctor". I'm _not_ here to debate medical neccessity. Only to posit that this situation _could_
exist and propose a way to handle it, mathematically and, perhaps, to tell an interesting story. Once again, these
aren't the opinions of my employer. 

_N.b.,_ and for this article in particular, I'm not here to debate medical neccessity. Only to posit that this situation _could_ exist and propose a way to handle it, mathematically and, perhaps, to tell an interesting story. As a data scientist, let's say your posed with the following problem prompt.


## Intro

### The problem
````{admonition} The prompt

Hello Data scientist! 
 
The team is creating a new program where the goal is to make sure that the right patient gets the right procedure or intervention at the right time. Assuming that some fraction of members currently destined to get a given procedure would have better health outcomes if they were to get a different procedure, how would you calculate the incremental potential cost and clinical benefit? 
````

### A note on the ethics of behavioral nudging
It's almost impossible to seperate helping a person to get the right procedure at the right time purely conceptually. In other words, it's difficult to seperate _how_ one might help a person do this from  

### Why this matters: some reductive clinical framing. 
I want to make sure that we solve for things _generically_, but I'll add some clinical framing because I think it helps the medicine go down, let's zoom back to the early 2010s. In particular, this report on `all things considered`
more than a decade
ago: [Surgery May Not Be The Answer To An Aching Back](https://www.npr.org/templates/story/story.php?storyId=125627307#:~:text=Unnecessary%20Back%20Surgery%20On%20The%20Rise%2C%20Study%20Says%20Too%20many,the%20benefit%20isn't%20there)
. 


As a reductive summary, this report and the associated literature pose a convincing argument that not all back surgeries are medically neccessary. When patients have a choice and are well educated, other non-surgical interventions may yield better
outcomes in some spheres. However, it's _very_ difficult to do 'nudge' fairly and impossible to evaluate
the [contrapositive](https://en.wikipedia.org/wiki/Contraposition). Unfortunately, the rabbit hole goes much deeper than just an NPR report and academic literature. There are literal _felons_ out there performing uneccessary procedures. In a particularly agregious case,
this [US obsetetrician was sentenced to ~60 years in jail for healthcare fraud](https://www.bmj.com/content/373/bmj.n1317)
. 

<p align="center">
   <iframe src="https://giphy.com/embed/fX8H6GACvaiatkqexi" width="480" height="279" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></p>
   <p align="center">While I reconcile that we're truly living in a dystopian nightmare, let's take some joy from one of the absolute worst, best movies from that late 90s clearly foreshadows this back-related mass balance crisis. <a href="https://giphy.com/gifs/dr-said-says-fX8H6GACvaiatkqexi">via GIPHY</a></p>

For our basic example, and using the NPR article as a basis, let's assume that let's assume that our membership exists in 4 states. These states describe the instantaneous membership destined for these procedures at a given time. This is, of course, a
simplification. However, let's suspend disbelieve and assume that these states represent the 'known universe':

1. Surgery. People destined to get surgery.
2. PT. People destined to get physical therapy.
3. Injections. People destined to get injections.
4. Self-care. People destined to 'suffer in silence' or, alternatively, actually can manage without a clinical
   intervention

As anyone currently destined for a given procedure could, theoretically, be best managed with another procedure, we'll
define the potential transitions between each state in the following manner (Figure Figure
{figure}`mass_balance_figure`).

```{glue:figure} mass_balance_figure
:name: "mass_balance_figure"

A simple, 4 box mass balance model. Arrows between the box represent rates, expressed in per time units as a normal series of linked differential equations would. 
```

The mechanisms at play can get complicated. 
1. Not everyone destined for a given procudure should get another procedure. 
2. Not everyone who is 'nudged' to a 'better' procedure will switch. 
3. Not everyone who is 'nudged' to a 'better' procudure will stay nudged e.g., 

### Why is it important to use a mass balances here? 

For this case, it's hard to estimate the impact of a program without considering the entire system of possible
interventions. As the web of shifts of procedures from one type to another type gets more complex, it becomes harder to keep track. 


This situation is _perfect_ for diffeqs and a mass balance approach. After all, and once again, we're __
not__ trying to stop people from getting procedures, we're trying to make sure that they get the right one.

Within this post, I'm going to attempt to do the following:

1. Parameterize the current state of procedures in matrix form.
2. Parameterize the flow of membership between procedures as a Markov~ish, mass-balance transfer matrix approach at
   steady state.
4. Draw some conclusions about the future states between Markov~ish (2) and PDE (3) estimates of the future state.

## Parameterizing the universe

### Systems of linear first-order differential equations.

If you don't know a little calculus, this is going to be really, really confusing. So let's start there. 

In our example, each compartment has an instantaneous number of members in it (`S`, `PT`, `I`, and `SC`) and rates of
flow in and out of each compartment (e.g., the flow from surgery, `S`, to PT, `PT`, is $f_{s-sc}$). Let's brute-force
describe each mass balance. I've probably missed a few of these (likely), but here's the description.

```{math}
:label: dSdt_eq
\begin{align}
\frac{d(S)}{dt} &= -S*f_{s-pt} - S * f_{s-sc} - S * f_{s-i} + PT * f_{pt-s} + I * f_{i-s} + SC * f_{sc-s}\\
\frac{d(PT)}{dt} &= S*f_{s-pt} - PT * f_{pt-s} - PT * f_{pt-i} - PT * f_{pt-sc} + SC * f_{sc-pt} + I * f_{i-pt}\\
\frac{d(I)}{dt} &=  S * f_{s-i} - I * f_{i-s} - I * f_{i-pt} - I * f_{i-sc} + SC * f_{sc-i} + PT * f_{pt-i}\\
\frac{d(SC)}{dt} &= S * f_{s-sc} - SC * f_{sc-s} - SC * f_{sc-pt} - SC * f_{sc-i} + I * f_{i-sc} + PT * f_{pt-sc}
\end{align}

```

If you assume that each of this is a first-order differential, you can put this in a simler matrix form by 'lumping up
terms'

```{math}
:label: dSdt_eq_reorg
\begin{align}
\frac{d(S)}{dt} &= \color{red}{S(-f_{s-pt} - f_{s-sc} - f_{s-i})} + PT * f_{pt-s} + I * f_{i-s} + SC * f_{sc-s}\\
\frac{d(PT)}{dt} &= S*f_{s-pt}  + \color{red}{PT(- f_{pt-s} - f_{pt-i} - f_{pt-sc})} + I * f_{i-pt} + SC * f_{sc-pt} \\
\frac{d(I)}{dt} &=  S * f_{s-i} + PT * f_{pt-i} + \color{red}{I (- f_{i-s} - f_{i-pt} - f_{i-sc})} + SC * f_{sc-i} \\
\frac{d(SC)}{dt} &= S * f_{s-sc} + PT * f_{pt-sc} + I * f_{i-sc}  + \color{red}{SC(-f_{sc-s} - f_{sc-pt} - f_{sc-i})} 
\end{align}
```

#### If your goal is to understand the system dynamics _generically_, take a detour to eigen town.

If your goal is to understand the dynamics of the system, analytically, I would suggest that you take a detour to eigen
town to solve for an analytical solution to this problem _generically_ (Most Elegant, but difficult to convey).
Afterall, and as an exceedingly clever data scientist, you'll have noticed that the above equations can be represented
in following form:

```{math}
\begin{equation}
\frac{d}{dt} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} = \begin{pmatrix} A & B & C & D \\ E & F &G &H \\I & J & K & L\\ M & N & O & P \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \\y_3 \\y_4 \end{pmatrix}
\end{equation}
```

This, in turn, can be posed generically as:

```{math}
\begin{equation}
dot(x) = A\cdot x
\end{equation}
```

Where you can then reposition it as the eigen value problem, where you can substitute in $e^{\lambda t}$ as x and then
solve.

```{math}
try: 
\begin{align}
x(t) &= \nu e^{\lambda t}\\
&\downarrow\\
\lambda \nu e^{\lambda t} &= A \nu e^{\lambda t}\\
&\downarrow\\
\lambda \nu &= A \nu 
\end{align}
```

Those of you who remember your matrix algebra $\lambda \nu = A \nu $ is
the [eigenvalue problem](https://www.sciencedirect.com/topics/mathematics/eigenvalue-problems). Where $\lambda$ are the
eigen values of matrix A and $\nu$ are the eigen vectors.

You can then compute the eigenvalues using the characteristic equation $det(A - \lambda I)=0$, which is a quadratic
formulation for lambda. For matrices larger than 2x2, it's super tedious to do this by hand. However, that's why
programming was invented, right? From here, you can generalize the solution to the problem in terms of the eigen values
and eigenvectors arbitrarily for any boundary condition and state. This is truly the _best_ way to solve it, although
it's tough to express this to your peers.

#### If your goal is to forecast a future state with known boundary and transition coefficients, go Markov~ish.

Notice that the $f$ values could represent either probability transfer (a Markov~ish definition) or kinetic first order
coefficients describing transfer between two states. If you want to simply project into the future to say somethin
like, "given my current state and my understanding of this program's potential affects, what is the future state?", then
you should/could do one of two things.

1. Go for a Markov~ish projection of the next state by parameterizing the transition matrix. This will not result in a
   time series of the program over time. However, it will allow facile 'forcasting' of the next state assuming whatever
   transitions you think the program is able to drive (or are appropriate).

2. Go for a time-resolved ODE approach. This method would be easy to show, but difficult to well parameterize and could
   lead to instability or weird results if your parameterization is off. However, it would give you a distinct time
   series projection of how the program would/could shape the membership.

For this, let's pull these out Equation {eq}`transfer_matrix` describes such a transfer state matrix describing the
rates of transfer for our 4 procedure simplification of the universe

```{math}
:label: transfer_matrix

\begin{equation}
\mathbf{T}_{transfer} = 
\begin{vmatrix} 
f_{s\rightarrow s} & f_{s\rightarrow i} & f_{s\rightarrow pt} & f_{s\rightarrow sc}\\
f_{i\rightarrow s} & f_{i\rightarrow i} & f_{i\rightarrow pt} & f_{i\rightarrow sc}\\
f_{pt\rightarrow s} & f_{pt\rightarrow i} & f_{pt\rightarrow pt} & f_{pt\rightarrow sc}\\
f_{sc\rightarrow s} & f_{sc\rightarrow i} & f_{sc\rightarrow pt} & f_{sc\rightarrow pt}\\
\end{vmatrix}
\end{equation}
```

Here, we assume that there exists a matrix describing the probability of transfer between two states. Along the diagnal
is the probability that the state _doesn't_ tranfer. In this case, this represents the static or immobile fraction of
members that do not transfer. Put in a more 'Markov~ish' framing for our example, this is the probability that a member
that is currently on a path for a given intervention _stays_ dedicated to getting this intervention.

The nice part about this approach is that you don't neccessarily need to parameterize the kinetics between states to be
able to project the next state with this approach. If you assume that, without intervention, the current state and the
future state will be ~roughly the same (which is typically a good assumption), then you can position this matrix to
represent how you _want_ to shape the future state by codifying the transition matrix as the intended program effect.
Let's take a simple case. Let's assume that you anticipate that you can shift ~3% of surgeries that should be
injections, 1% of injections that should be surgeries, and 5% of folks getting 'self care' to get injections, and

1.

####   

The key is to condition a transfer matrix that describes the relative transfer rate of one type or compartment to
another type or compartment.

```{admonition} Check your math!
:class: tip
Note that each column will sum to 1. This matrix should be interpreted in terms of the fraction of `row` that flows into `column
```

The current state will also need to be represented in matrix notation, where each element represents the current number
of cases / visits for each category.

```{math}
:label: curstate_mat
\begin{equation}
\mathbf{State}_{current} = 
\begin{vmatrix}
\mathbf{N_{surgery}}  \\
\mathbf{N_{injection}}  \\
\mathbf{N_{pt}}  \\
\mathbf{N_{selfcare}}  \\
\end{vmatrix}
\end{equation}
```

Let's assume that the current state looks like this: 10,000 Surgeries, 4,535 Injetions, 45,500 PT episodes, and 0
self-care episodes.

```
current_state = np.matrix([[10000],[4535],[45500],[0]])
```

```{glue:} mass_bal_current_state
```

Let's then define the transfer matrix, as follows. Note that the definitions of the fractions in the matrix are
equivalent to behavior change rates in the off-axis. For the eye of the matrix, these values represent the fraction of
immobile procedures. Further note that defining these value can be tricky to interpret. If the eye of the matrix is 1,
then all values in the column should be, by definition, 0. This does not mean that there's no transfer into that
procedure column or row. It simply means that all procedures that are routed there, stay there.

```
transfer_matrix = np.array([[0.76,0,0,0],
                            [0.1128,0.61,0,0],
                            [0.0072,0,1,0],
                            [0.12,0.39,0,1]]))
```

```{glue:} mass_bal_transfer_mat
```

We can solve the future state using the flow matrix F multiplied by the current state matrix X.

```{math}
:label: mass_bal_future_state

\begin{equation}
\mathbf{State}_{future} = \mathbf{T}_{transfer} \cdot \mathbf{State}_{current}
\end{equation}
```

```
future_state = transfer_matrix*current_state
```

```{glue:} mass_bal_future_state
```

Amaazing! Now that we have the future state of procedures, it's simple to find the future-current state and multiply it
by the cost matrix to determine the MCS. Let's define the cost matrix as follows.

```
cost_matrix = np.matrix([[15000],[500],[1000],[5]])
```

```{glue:} cost_matrix
```

Then let's determine the MCS with Equation {eq}`mcs`.

```{math}
:label: mcs

\begin{equation}
\mathbf{MCS} = (\mathbf{State}_{future} - \mathbf{State}_{current}) * \mathbf{Cost}
\end{equation}
```

```
mass_bal_mcs = (future_state - current_state).ravel()*cost_matrix
```

```{glue:} mass_bal_mcs
```

Congrats, we've saved a lot of money!

```{warning}
Please note that these behavior change assumptions are quite, quite large. In reality, it's very unlikely that we could 'nudge' a significant portion of members from surgery into other interventions. However small, or large, our behavior change, it's critical to correctly account for the procedures in a mass balance compliant means as not to over or under calculate med cost savings. 
```

2.

. This is, in turn, functionally a mass balance in that the relationship between the current and future state is defined
by a state matrix.

Taking one intervention as an example, let's describe the flow of people in, and out, of that box as a differential
equation or  'mass balance'. Equation {eq}`dSdt_eq` shows an example mass balance of Surgeries.

In this equation, we describe the change in surgeries over time $\frac{dS}{dt}$, as well as all the kinetic shifts in
and out of the

### State space - the markov chain~ish transition matrix.

Markov chains are systems in which the relationship between the current state and the future state are rigorously
defined in terms of probabilities. Here, we're trying to understand the impact of our program designed, once again, to
make sure that the _right_ people get the _right_ procedure or intervention at the _right_ time.

Let's start with a single example that can be extrapolated out. For our example, let's draw a theoretical boundary
around a given procedure: Surgeries.

and

In the steady state, stable case, $\frac{dS}{dt}=0$. Thus, you can re-write the equation as follows.

```{math}
:label: dSdt_eq2
\begin{equation}
0 = -S*f_{s-pt} - S * f_{s-sc} - S * f_{s-i} + PT * f_{pt-s} + I * f_{i-s} + SC * f_{sc-s}
\end{equation}
```

Because we're awesome, let's define the current membership that exists within these states as a matrix.

```{math}
:label: massbalance-transfer-matrix-eq1
\mathbf{S}_{0} = 
\begin{vmatrix}
\mathbf{N_{surgery}}  \\
\mathbf{N_{injection}}  \\
\mathbf{N_{pt}}  \\
\mathbf{N_{selfcare}}  \\
\end{vmatrix}
```

If one is patient, you can do this for every single 'box', coming up with a series of linked mass balance described by
differential equations. These differential equations can then be solved simultaneously for steady state using a matrix
method. 


