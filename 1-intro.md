## Down the Wabbit Hole

Some Haskell Design Patterns in ML Engineering

Chris McKinlay

<img id="plain" src="formation.png" style="max-height: 100px; ">

Note:

- Hi! Hi Im Chris and I'm a principal engineer at Formation, where I work mostly on on ML stuff. 
- We're using reinforcement learning to reinvent customer loyalty programs.
- I'm going to share some best practices we've been using in our Haskell codebase to help us write better ML code.


# ML best practices?
## AKA
<!-- .element: class="fragment" -->
## my opinion
<!-- .element: class="fragment" -->
Note: 
On many issues I dont think theres an established consensus. 


## What are the criteria?

* Correctness?
<!-- .element: class="fragment" -->
* Performance?
<!-- .element: class="fragment" -->
* Testability?
<!-- .element: class="fragment" -->
* Maintainability
<!-- .element: class="fragment" -->

Note:

- Correctness can be difficult or impossible to define for ML systems.
- Performance of course is important, but it's not the top priority. 
- Testability is a really good metric for ML code! But it's downstream of something more general.
- Which is maintainability or adaptability. We need software we can maintain, refactor, and modify easily. IMO this is the most important thing, esp for a small startup. 
- As I mentioned, we're trying to apply reinforcement learning to customer relationships. Assumptions are bound to change.
- Earlier this year we had a maintianability problem. 


# 
![](vw.png) 
Note:
- at the time, we had identified a simplified version of our RL problem that we thought we could solve with contextual bandits
- we opted to use VW,  an out-of-box SGD library for solving CB problems that our data scientists understood how to use
- the problem was that the code that had been built up around VW was overly specific to VW, and it had started to affect other systems


# 
![](high-interest.png) 
Note:
- there's a well-known paper form google that address exactly this issue, which they term the 'glue code design pattern'
- they define this as a massive amounts of supporting code written to get data into and out of general-purpose packages.


## Glue Code

> "Glue code is costly in the long term because it tends to freeze a system to the peculiarities of a specific package."

Note:
- so glue code makes software difficult to adapt and maintain
- the paper does provide a recommendation for avoiding glue code


# 
> "Glue code can be reduced by choosing to re-implement specific algorithms within the broader system architecture."
<img id="plain" src="blinkingman.gif" style="max-height: 2000px; ">
<!-- .element: class="fragment" -->
Note:
- i dont think this advice is very practical. 
- i mean maybe it is if you're Google and have lots of people to throw at a problem.
- but for anything moderately complex this is terrible advice for a startup.
- the functionality that vw provides is far too complex to reimplement quickly, and even if we did we had very little assurance that our use case would be the same when we finished


## Interfaces FTW!

Note:
- solution has been to establish common interfaces to abstract away vw and other packages 
- allows supporting code to be more reusable and reduces the cost of changing packages.
- in this talk i'll explain how we use haskell's type system and some design patterns to accomplish this
- show how it works w/ tensorflow and vw
- finally go through a use case with some supporting library code


