# B: mask and bracket


</aside>
</script></section>
<section data-markdown><script type="text/template">
Example: Learn and Predict

```haskell
tryToLearn :: Session -> ByteString -> IO (Vector Float)
tryToLearn ses bs = do
    ex <- readExample ses bs
    _ <- learn ses ex
    _ <- predict ses ex
    res <- getMulticlassProbsPrediction ex
    finishExample ses ex
    return res
```

<aside class="notes">
So here's the basic workflow i was trying to use in the Repl. Just exposing this type of function and hiding readExample and finishExample from the user will care of a lot of problems.

What if  `learn`, `predict` etc throw an exception? I'll still be leaking space b/c i'll have failed to call finishExample. 
</aside>
</script></section>
<section data-markdown><script type="text/template">

```haskell
tryToLearn' :: Session -> ByteString -> IO (Vector Float)
tryToLearn' ses bs = do
    ex <- readExample ses bs
    let learnPredict = learn ses ex *> 
                       predict ses ex *> 
                       getMulticlassProbsPrediction ex
    eres <- try $ learnPredict
    finishExample ses ex
    case eres of
        Left ( e :: VWError ) -> throwIO e
        Right res -> return res

try :: IO a -> IO (Either IOException a)
throwIO :: IOException -> IO a
```
<aside class="notes">
Let's add some basic exception handling with try and throwIO from the Control.Exception module.

I'm also going to add some type annotations so ghc knows what kind of  need to add some type annotations here or ghc will complain. WERK
</aside>
</script></section>
<section data-markdown><script type="text/template">
Factor out the use case:

```haskell
withEx :: Session -> ByteString -> (Example -> IO a) -> IO a
withEx ses bs action = do
    ex <- readExample ses bs
    eres <- try (action ex)
    finishExample ses ex
    case eres of
        Left ( e :: VWError ) -> throwIO e
        Right res -> return res
```
</script></section>
<section data-markdown><script type="text/template">

```haskell
tryToLearn'' :: Session -> ByteString -> IO (Vector Float)
tryToLearn'' ses bs = withEx ses bs (learnPredict ses)

learnPredict :: Session -> Example -> IO (Vector Float)
learnPredict ses e = learn ses e *>
                     predict ses e *> 
                     getMulticlassProbsPrediction e
```
<aside class="notes">
Now we can write tryToLearn using the higher order function withEx and tha action learnPredict

General principle: Avoid using functions which only allocate or only clean up whenever possible.

So is that enough to take care of exceptions?

</aside>
</script></section>
<section data-markdown><script type="text/template">

That's enough... to handle synchronous exceptions.

Synchronous exceptions are exceptions which are generated directly
from the `IO` actions you are calling.

</script></section>
<section data-markdown><script type="text/template">
Asynchronous Exceptions (Courtesy of M. Snoyman)

* are exceptions thrown from another thread
* occur just like synchronous exceptions
* are caught with the same handlers (e.g. `try`)
* can happen at unexpected times!

</script></section>
<section data-markdown><script type="text/template">
The need for masking

```haskell
withEx :: Session -> ByteString -> (Example -> IO a) -> IO a
withEx ses bs action = do
    checkAsync -- 1
    ex <- readExample ses bs
    checkAsync -- 2
    eres <- try (action ex)
    checkAsync -- 3
    finishExample ses ex
    checkAsync -- 4
    case eres of
        Left ( e :: VWError ) -> throwIO e
        Right res -> return res
```

1 or 4: fine, 2 or 3: not fine
<aside class="notes">
Let's revisit our `withEx`, with explicit async-exception checking calls
</aside>
</script></section>
<section data-markdown><script type="text/template">
The `mask_` function

```haskell
mask_ :: IO a -> IO a

withEx ses bs action = mask_ $ do ...
```

* Fixes the resource leak.
* Necessary b/c this library is going to be used in a multithreaded application.
* Recall we also used this in `newSession`.

<aside class="notes">
So are we good ... now?
</aside>
</script></section>
<section data-markdown><script type="text/template">

* But now other threads can't kill `action`!
* This is bad if `action` is an expensive computation.
* It's good practice to restore the previous masking state inside function arguments.

</script></section>
<section data-markdown><script type="text/template">
The `mask` function

Like `mask_`, but with a function to temporarily restore 

```haskell
mask :: ((forall a. IO a -> IO a) -> IO b) -> IO b
mask io = ...

mask_ :: IO a -> IO a
mask_ io = mask $ \_ -> io
```
<aside class="notes">
mask has a pretty interesting type signature, so let's look at a typical use case.
</aside>
</script></section>
<section data-markdown><script type="text/template">
Restoring unmasked state

```haskell
withEx :: Session -> ByteString -> (Example -> IO a) -> IO a
withEx ses bs action = mask $ \restore -> do
    ex <- readExample ses bs
    eres <- try (restore (action ex))
    finishExample ses ex
    case eres of
        Left ( e :: VWError ) -> throwIO e
        Right res -> return res

mask :: ((forall a. IO a -> IO a) -> IO b) -> IO b
```
<aside class="notes">
What is the type of restore here? forall a. IO a -> IO a

What does that existensial quantifier mean? (only can do IO () type stuff) Why is it there?  

Safe to `restore` unmasked state here, because it's wrapped in a `try`

</aside>
</script></section>
<section data-markdown><script type="text/template">
Two classes of handlers

```haskell
-- for recovery (rethrows immediately on async)
try :: Exception e => IO a -> IO (Either e a)
catch :: Exception e => IO a -> (e -> IO a) -> IO a

-- for cleanup (always rethrows after running cleanup)
onException :: IO a -> IO b -> IO a
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
```
<aside class="notes">
If you want to do some cleanup in the event that an exception is raised, use finally, bracket or onException.
To recover after an exception and do something else, the best choice is to use one of the try family.
... unless you are recovering from an asynchronous exception, in which case use catch or catchJust.
The difference between using try and catch for recovery is that in catch the handler is inside an implicit mask 
</aside>
</script></section>
<section data-markdown><script type="text/template">
The bracket pattern
```haskell
-- Control.Exception
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
bracket before after action =
    mask $ \restore -> do
        a <- before
        r <- restore (action a) `onException` after a
        _ <- after a
        return r
```
<aside class="notes">
B/c we're more interested in cleanup than in doing any sort of recovery, we can go ahead and use the bracket pattern. 

onException only performs the final action if there was an exception raised by the computation.

</aside>
</script></section>
<section data-markdown><script type="text/template">
Applying the bracket pattern

```haskell
import Control.Exception (bracket)

withExample :: Session 
            -> ByteString 
            -> (Example -> IO a)
            -> IO a
withExample ses bs = 
    bracket (readExample ses line) (finishExample ses)
```
</script></section>
<section data-markdown><script type="text/template">
Applying the Bracket Pattern

```haskell
withUnsafeVW ::
       ByteString
    -> LabelType
    -> PredictType
    -> (Session -> IO r)
    -> IO r
withUnsafeVW opts ltype ptype action =
    bracket (newUnsafeVW opts ltype ptype) deleteSession action
```
<aside class="notes">
Finally, we can go one step further and apply bracketing to VW contexts as well.
</aside>
</script></section>
</section>

