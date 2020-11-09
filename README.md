# PrEd-graphdrawing
An implementation of the PrEd graph drawing algorithm in `python`, with a few improvements from ImPrEd.

## Planar Graph Drawing

The aim of this package is to draw a planar graph in an aesthetically pleasing manner, while preserving the original number of edge intersections. To do this, it uses a force-directed method similar to the Kamada-Kawai method, but sets the maximum distance a node can move in each step.

The forces on each node are made up of the following:
* An attraction between neighbouring nodes
* A repulsion between all pairs of nodes
* A repulsion between each node and each edge.

## ImPrEd improvements

This implementation takes a few improvements from ImPrEd. Importantly, it uses a bounding edge computation to reduce the number of edges that must be calculated. A bounding edge is one on the boundary of any face containing a node $ v $. 
_note that the ImPrEd paper can do this computation for non-planar graphs as well. That is currently unimplemented._
This boundary is calculated using `networkx`'s immensely handy `traverse_face` function.

We also apply a gravity force to the nodes. This attracts each node linearly to the centre of gravity of the graph, and leads to nicer convexity properties.

## Limitations
This implementation currently has a few limitations.
Those limitations are:
* Only works with planar graphs. This is a limitation of the bounding edges calculation, and could be lifted.
* Cannot draw graphs where any two edges are collinear. It's not clear in this circumstance how to define a boundary to conserve.

# Demonstration usage
```
import networkx as nx
from pred import PREDLayout

G = nx.wheel_graph(10)
pos = PREDLayout(G).anneal(steps=100)
nx.draw(G, pos=pos)
```

# References

[1] A Force-Directed Algorithm that Preserves Edge Crossing Properties, Francois Bertault, Information Processing Letters 74, 1–2 (Apr. 2000), 7–13.
[2] ImPrEd: An Improved Force‐Directed Algorithm that Prevents Nodes from Crossing Edges.  Simonetto, P., Archambault, D., Auber, D. and Bourqui, R. (2011), Computer Graphics Forum, 30: 1071-1080. https://doi.org/10.1111/j.1467-8659.2011.01956.x
