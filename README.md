# MICE
This project is a personal research endeavor that remains a work in progress. I extend my apologies for any errors you may come across during your exploration.

## Idea
Sensitive analysis is an important tool for understanding the behavior of complex systems. In this article, we propose a novel framework for simultaneous learning of copula simulation and density estimation in a non-parametric way, enabling both global and local sensitive analysis with a single calibration of the model.

- We introduce a framework that allows for simultaneous learning of copula simulation and density estimation in a non-parametric manner, which is the first of its kind.
- We use copulas to perform both global and local sensitivity analysis with only one model calibration, and we define a local mutual information index.
- We compare our method to state-of-the-art standards for sensitivity analysis.
  
## Results 

We conducted a comparison between two models - the first model, MICE, trains CODINE directly on the simulation data of IGC, while the second model trains IGC and CODINE separately. Our results in the next section show that MICE can reduce the variance of CODINE’s learning, improve its performance slightly, and is much more suitable for interpretability of the framework as a whole.

![Alt Text](https://github.com/c1adrien/MICE/blob/main/results%20paper/model.png)


Our model has successfully learned the copulas and density levels. By observing the imperfections, we can have a good idea of the quality of our estimation and anticipate the model’s performance. Then it is easy to compute mutual information coefficient. 


![Alt Text](https://github.com/c1adrien/MICE/blob/main/results%20paper/copulas_learned.png)


Feel free to explore our code and documentation to learn more about this approach and its practical implications.
