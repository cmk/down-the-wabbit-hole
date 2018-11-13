# A: Reader and Cont

Note:


# 
```haskell
newtype Cont r a = Cont { runCont :: (a -> r) -> r }

ghci> :t flip ($) True
flip ($) True :: (Bool -> b) -> b
ghci> runCont (ack 3 4) id
125
```
<aside class="notes">
</aside>
</script></section>
<section data-markdown><script type="text/template">
<aside class="notes">	'
</aside>
</script></section>
<section data-markdown><script type="text/template">
Cont is a Functor

```haskell
instance Functor (Cont r) where
  fmap ab (Cont arr) = Cont $ \br -> arr (\a -> br (ab a))

-- Factorial 
fact :: Int -> Cont Int Int
fact 0 = Cont ($ 1)
fact n = (n*) <$> fact (n-1)
```
</script></section>
<section data-markdown><script type="text/template">
(>>=) is just function application.

```haskell
instance Monad (Cont r) where
  return = Cont $ \k -> k a --pure
  (Cont arr) >>= acb = Cont $ \br -> arr (\a -> runCont (acb a) br)
```
<aside class="notes">
So, replacing (a -> b) -> b with Cont b a, what's the monadic type for our basic building block of reverse function application? 

a -> (a -> b) -> b translates to a -> Cont b a... the same type signature as return and, in fact, that's exactly what it is.
</aside>
</script></section>
<section data-id="9843e35ec7d08106b08ebf334b9a6b89"><div class="sl-block" data-block-type="image" style="width: 1064px; height: auto; left: 166px; top: 140px; min-width: 4px; min-height: 4px;" data-block-id="e4d55e127130b3ed60349cfc422d2289"><div class="sl-block-content" style="z-index: 11;"><img style="" data-natural-width="2128" data-natural-height="1350" data-lazy-loaded="" data-src="ackermann.png"></div></div></section>
<section data-markdown><script type="text/template">
Ackermann Function

```haskell
ack :: Int -> Int -> Cont Int Int
ack 0 n = pure (n+1)
ack m 0 = ack (m-1) 1
ack m n = do x <- ack m (n-1)
             y <- ack (m-1) x
             return y

runCont (ack 3 4) id --125
```
<aside class="notes">
Let's break down the mechanics of this with a more familiar function.
</aside>
</script></section>
<section data-markdown><script type="text/template">
Fibonacci Function

```haskell
fib' 0 c = c 0
fib' 1 c = c 1
fib' n c = fib' (n-1) d
           where d x = fib' (n-2) e
                 where e y = c (x+y)

```
<aside class="notes">
Imagine you have a machine without a call stack - it only allows tail recursion. How to execute fib on that machine? You could easily rewrite the function to work in linear, instead of exponential time, but that requires tiny bit of insight and is not mechanical.

The obstacle to making it tail recursive is the third line, where there are two recursive calls. We can only make a single call, which also has to give the result. Here's where continuations enter.
</aside>
</script></section>
<section data-markdown><script type="text/template">

Equivalently, we can use lambdas:

```haskell
fib' 0 = \c -> c 0
fib' 1 = \c -> c 1
fib' n = \c -> fib' (n-1) $ \x ->
                 fib' (n-2) $ \y ->
                   c (x+y)
```
The last three lines smell like a do block...
</script></section>
<section data-markdown><script type="text/template">


```haskell
fib' 0 = return 0
fib' 1 = return 1
fib' n = do x <- fib' (n-1)
            y <- fib' (n-2)
            return (x+y)
```
</script></section>
<section data-markdown><script type="text/template">
Fibonacci is inherently applicative:

```haskell
fib :: Int -> Cont Int Int
fib 0 = pure 0
fib 1 = pure 1
fib n = (+) <$> fib (n-1) <*> fib (n-2)
```

</script></section>
<section data-markdown><script type="text/template">
```haskell
instance Applicative (Cont r) where
  pure a = Cont ($ a) --Cont $ \k -> k a
  (Cont abrr) <*> (Cont arr) = Cont $ \br -> abrr (\ab -> arr (\a -> br $ ab a))
```

</script></section>
<section data-markdown><script type="text/template">
Cont objects can be chained together, so that the continuation you pass in threads through the guts of all the Cont objects in the chain before it's finally invoked. The way they chain is the way Cont works: each object in the chain invokes a continuation that has the next object's computation prepended to the final continuation. Let's say we have a chain of Cont objects f1 -> f2 -> f3, and let's say you had a continuation c3 that you want to pass to the chain. Then:
f3 needs to invoke c3 when it's done.
f2 needs to invoke a continuation c2 that will invoke f3, which will invoke c3.
f1 needs to invoke a continuation c1,
 which will invoke f2,
  which will invoke c2,
   which will invoke f3,
    which will finally invoke c3.
To chain the Cont objects together, then, we need to create the appropriate continuation functions c1 and c2 and make sure they get passed as the continuation argument to f1 and f2 respectively.
<aside class="notes">	'
</aside>
</script></section>
<section data-markdown><script type="text/template">
Unit test in our Tensorflow Serving code

```haskell
withPredictionClient :: PredictionClientConfig -> (PredictionClient -> IO r) -> IO r
withPredictionClient pcc f =
  G.withGRPC $ \grpc ->
    G.withClient grpc TFS.PredictionService (mkClientConfig pcc) $ \psc ->
      G.withClient grpc TFS.ModelService (mkClientConfig pcc) $ \msc ->
        f $ mkPredictionClient pcc psc msc

withPredictionClient :: PredictionClientConfig -> (PredictionClient -> IO r) -> IO r
withPredictionClient pcc f = do
  grpc <- G.withGRPC 
  psc <- G.withClient grpc TFS.PredictionService (mkClientConfig pcc) 
  msc <- G.withClient grpc TFS.ModelService (mkClientConfig pcc) 
  return $ f (mkPredictionClient pcc psc msc)
```

