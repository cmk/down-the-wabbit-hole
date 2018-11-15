# The Handle Pattern

Note:

Haskell design patterns are a blogosphere phenom, so I'll include some links at the end of the talk.
The ones you've heard about are probably beautiful and mathematically sound. My colleague Greg Phiel is giving a talk on that kind of thing on Saturday that you should check out. 

However w/ bindings to ML libraries we are dealing with inherently mutable state, and external code which doesn’t behave nicely.

In those cases, we need another approach. What we’re going to describe feels a bit similar to Object Oriented Programming.


# 
* Encapsulating and hiding state inside objects
<!-- .element: class="fragment" -->
* Providing methods to manipulate this state rather than touching it directly
<!-- .element: class="fragment" -->
* Providing interfaces for configuration, creation, and destruction of objects
<!-- .element: class="fragment" -->
Note:
Pretty much any sort of Haskell code can be written in this particular way, but that doesn’t mean that you should. 
This method relies heavily on IO. 
Whenever you can write things in a pure way, you should attempt to do that and avoid IO. 
This pattern is only useful when IO is required


# A Simple Handle

```haskell
module Policy.Source.Types where

data SourceHandle = SourceHandle {
    loadModel :: ModelName -> 
                 Maybe ModelVersion -> 
                 IO (Either SourceError ModelFile)
  , saveModel :: ModelName -> 
                 ModelFile -> 
                 IO (Either SourceError ModelVersion)
  , listModelVersions :: ModelName -> 
                         IO (Either SourceError [ModelVersion])
}
```
Note:
This is a handle that represents a model location, either on the local filesystem, S3, a database,etc.
We have 3 functions.
We use TFS at work to serve a wide variety of models. 
TFS expects a model to consist of a directory w/ subdirs for each version. 
So we tend to use this format everywhere. 
This handle needs to do 2 things behind the scenes: 
- manage scare resources efficiently
- ensure version integrity


# Polymorphic Handles

```haskell
module Policy.Model.Types where

data ModelHandle i j o e m = ModelHandle {
     learn :: i -> m ()
   , score :: j -> m o
   , error :: i -> m e 
}
```
<!-- .element: class="fragment" -->
Note:
Here is a 
A MonadIO constraint is typically placed on m by callers.
Separate training and serving inputs. 
Used for both supervised learning (cust embeddings) and rl
Error type is used to drive early stopping behavior w SGD.


#  

```haskell
{-# LANGUAGE TypeFamilies #-}                      -- 1
module Policy.Model.Types where

class ModelConfig c where
  type LInput c                             
  type SInput c
  type Output c
  type Error c
  type Finalizer c

  createModelHandle :: (MonadIO m, MonadMask m)    -- 2
                    => c 
                    -> m (ModelHandle' c m, Finalizer c)
  deleteModelHandle :: (MonadIO m, MonadMask m) 
                    => Finalizer c -> m ()

type ModelHandle' c m = 
ModelHandle (LInput c) (SInput c) (Output c) (Error c) m 
```

Note: 
here is the typeclass that we use to manage handles. basically a handle factory.
DESCRIBE create returns a tuple of the handle itself as well as a destructor
it uses type families. which provide a limited form of type-level functions in haskell 
so the five types listed there depend on the implementing type c.
so create model will specialize the type of modelhandle' and finalizer
- here are those constraints i mentioned earlier. the monadmask constraint on the handle provides for async exception handling. all implementations of this interface involve bindings to C or C++ libraries, 
- finally we provide a simple type alias to hide the dependant typing. 


#  

```haskell
{-# LANGUAGE TypeApplications #-}                  
module Policy.Model.API where 

import qualified Control.Exception.Safe as Safe

withModelConfig ::
  forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => c
  -> (ModelHandle' c -> m r)
  -> m r
withModelConfig c act =
  Safe.bracket (createModelHandle c)               
               (deleteModelHandle @c . snd)        -- 2 
               (act . fst)
```
```haskell
bracket :: forall m a b c . MonadMask m =>         -- 1
           m a -> (a -> m b) -> (a -> m c) -> m c
```
<!-- .element: class="fragment" -->
Note: 
heres an example from the api.
we feed it a configuration and a handle action, and it creates the handle, performs the action, and deletes the handle in a resource-safe fashion.
- uses what's referred to as the bracket pattern in haskell. like try / finally or with in python. 
  the a type here is (ModelHandle' c m, Finalizer c)
- the other lang extension we use a lot is called type applications. lets me pass type information explicitly. 
- also note that i'm grabbing the finalizer 
if i talk fast we'll see this in use in the last slide of the talk
