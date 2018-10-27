# Down the Wabbit Hole

## Some FP Design Patterns in ML engineering

Chris McKinlay


Note:

Hi! Hi Im Chris and I work at Formation, mostly on on ML stuff. 

We're trying to reinvent loyalty programs with reinforcement learning.

I'm going to share some techniques we've been using in our Haskell codebase to help us write better code.


# Better code?

* Correctness?
<!-- .element: class="fragment" -->
* Maintainability?
<!-- .element: class="fragment" -->
* Testability?
<!-- .element: class="fragment" -->
* Performance?
<!-- .element: class="fragment" -->

Note:

Of course, better is subjective. Faster code is better, but performance isn't
free -- you have to spend time implementing it.  Code that gives the right
answer is important, but sometimes "close enough" is good enough.

Maintainability is another good point. How easy is the software to maintain,
refactor, and modify? If busienss rules need to change, then how difficult will
it be to make these logical changes? Unfortunately, this is difficult to
understand without actually attempting to do the change, so it's often too
late.

Testability -- this is a really good metric for good code!
Testing code is often the first time that you go to actually use the code you wrote.
It's the first time you have to setup the surrounding code and infrastructure, and you get immediate feedback on how good your code is to reuse.


# Two common issues w/ ML code
## Reproducibility
![](kaggle-ML.png) <!-- .element: id="plain" -->

## Glue code
<!-- .element: class="fragment" -->

Note:

Machine learning packages may often be treated as black boxes, resulting in large masses of “glue code” or calibration layers that can lock in assumptions.

This is certainly true of vowpal wabbit, a contextual bandits library we bootstrapped with. Even with frameworks like tensorflow ...

Machine learning researchers tend to develop general purpose solutions as self-contained packages.


Using self-contained solutions often results in a glue code system design pattern, in which a massive amount of supporting code is written to get data into and out of general-purpose packages.

This glue code design pattern can be costly in the long term, as it tends to freeze a system to the
peculiarities of a specific package. 

General purpose solutions often have different design goals: they seek to provide one learning system to solve many problems, but many practical software systems are highly engineered to apply to one large-scale problem, for which many experimental solutions
are sought. While generic systems might make it possible to interchange optimization algorithms,
it is quite often refactoring of the construction of the problem space which yields the most benefit
to mature systems. The glue code pattern implicitly embeds this construction in supporting code
instead of in principally designed components. As a result, the glue code pattern often makes exper5
imentation with other machine learning approaches prohibitively expensive, resulting in an ongoing
tax on innovation.


Glue code can be reduced by choosing to re-implement specific algorithms within the broader system
architecture. At first, this may seem like a high cost to pay—re-implementing a machine learning
package in C++ or Java that is already available in R or matlab, for example, may appear to be
a waste of effort. But the resulting system may require dramatically less glue code to integrate in
the overall system, be easier to test, be easier to maintain, and be better designed to allow alternate
approaches to be plugged in and empirically tested. Problem-specific machine learning code can
also be tweaked with problem-specific knowledge that is hard to support in general packages.
It may be surprising to the academic community to know that only a tiny fraction of the code in
many machine learning systems is actually doing “machine learning”. When we recognize that a
mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code,
reimplementation rather than reuse of a clumsy API looks like a much better strategy


# Test Driven Design

* Informs API/library design
<!-- .element: class="fragment" -->
* Immediate feedback on code reusability
<!-- .element: class="fragment" -->
* Free regression/unit testing
<!-- .element: class="fragment" -->

Note:

Writing tests first is a great proxy for writing good code.
You get immediate feedback on your library design.
Having tests to cover correctness and find bugs is just about a nice side effect. 


# TDD is hard!

Note:

Writing test-first is *really* difficult.
This is partially because writing code that is nice to test (and, by proxy, reuse/modify/understand/etc) is really difficult.


# This isn't a TDD talk

Note:

I don't care about TDD. Testability is a nice *proxy* for "good" code.


<!-- .slide: data-background="finger-moon.jpg" -->
Note:

Code that's easy to test tends to be easy to modify, update, reuse, and verify.
It is like a finger pointing at the moon of good software: don't look at the
tests, look at the good software! You don't have to write tests to write good
code, and just because you wrote tests doesn't mean your code is nice.
