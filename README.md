# Riemannian metric in latent space

## Based on
```
Fast Approximate Geodesics for Deep Generative Models
Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, Patrick van der Smagt
```

## Summary
VAEs and other latent variable models learn lower dimensional manifolds of the data. Often one takes the lower dimensional representation of the data to do some further analysis, e.g. clustering.
However, if the learned manifold has curvature, Euclidean distance in latent space (i.e. straight lines) will not be a good measure of similarity for datapoints on the manifold.
Instead one should calculate distances along the geodesics of the manifold, but this is quite computationally intensive. In the paper above, the authors propose calculating the geodesic distance approximately (but faster):
- create a nearest neighbor graph in latent space, using euclidean distance. For short distances euclidean distance should still be an OK proxy for geodesics.
- For each edge in the nearest neighbor graph, calculate the actual geodesic distance. This will correct for nearest neighbor pairs that are in a region of high curvature (i.e. they should actually be further apart then they are by euclidean distance)
- Now, to approximate the geodesic distance between arbitrary points, we use shortest-path algorithms on the constructed graph (with edge distances being the riemannian distances).

## TODO
- toy example
- Optimize the computations: calculating the riemannian distance (`RiemannianMetric.riemannian_distance_along_line()`) is pretty slow due to tensorflow's way of calculating gradients.
