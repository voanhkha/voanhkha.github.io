---
layout: post
comments: true
title: (Rank 17/1620) Using Gurobi with custom handles 
excerpt: "and compare with CPLEX, to solve an optimization problem in Kaggle Santa 2019!"
mathjax: true
date:   2020-01-15 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
preview_media: /assets/previews/gurobi.jpg
preview_type: img   # gif | video
---

[Code 77777.77](https://www.kaggle.com/code/khahuras/the-elegant-prize)
[Solution write-up](https://www.kaggle.com/competitions/santa-workshop-tour-2019/discussion/122290)

Today, I have some quick tests with both CPLEX and Gurobi. Both are very different in terms of the API (how to define variables, add constraints…). Here are my thoughts:

1) They are equally convenient in coding (using Python API)
Suppose we have 5000x100 variables x[f, d] with f in [0, …, 4999] and d in [0, 99] indicating the binary variable of day d assigned to family f. Hence, we will have 5000 constraints: sum(x[f, d]) == 1 (with all d at a specific f).

In Gurobi, we can define the variables like this:
x = model.addVars(5000, 100, vtype=GRB.BINARY, name='x', ub=x_ub, lb=x_lb )

That means whatever the first arguments are put into addVars, they will automatically stacked into an n-dimensional variables, which later any variable x can be accessed as x[f, d]. In this case, we created a variable x with 5000 indices on the first axis and 100 indices on the second axis. That means we can define another set of y variables with 3 dimensions as y = model.addVars(100, 200, 300, vtype=..., ...). As a consequence, upper and lower bounds for x can be set by any same-size n-dimensional Python list of lists.

The 5000 constraints as described earlier, can be added by
model.addConstrs((x.sum(f ,'*') == 1 for f in range(5000)))

This is the compact syntax provided by Gurobi. However one can use this equivalent Pythonic syntax
model.addConstrs((sum([ x[f, d] for d in range(100) ]) == 1 for f in range(5000)))

The objective, for example, can be defined as
model.setObjective( sum([x[f, d]*PREF[f][d] for f in range(5000) for d in range(100)]) )

In CPLEX, we can define the variables as

x = mdl.binary_var_matrix(range(5000), range(100), "x")
preference_penalty = mdl.sum(x[f, d] * PREF[f, d] for f in range(5000) for d in range(100))
mdl.minimize(preference_penalty )

As shown, both can be defined nicely! At the first sight, I was leaning towards Gurobi, because the documentation is more accessible, and I was mistakenly using CPLEX's old Python API. The problem is that after installing CPLEX (using the installation from its official website), the installer guides me to install the deprecated Python API! With @cpmpml 's suggestion, I later found out that CPLEX's docplex is much better. But how can I miss docplex in the first place? (maybe my mistake indeed).

To this point, they are still tied :)

2) ….. is faster than ….. in solving
By the time of this writing, I am testing the solving speed of them, given the same formulation and initial solution.
Not sure if my CPLEX formulation is missed something, or CPLEX presolving process has something wrong, the beginning gap of CPLEX is 1.50% and decreasing slowly, while the beginning gap of Gurobi is 1.22% and decreasing quicker. This means CPLEX is significantly slower, but I am not sure, the difference is too big, so I may have a bug in CPLEX formulation. Not everything is sure now at this stage…

I will update this soon.
