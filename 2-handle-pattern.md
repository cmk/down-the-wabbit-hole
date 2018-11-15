# The Handle Pattern...

Note:

- Haskell design patterns are a blogosphere phenom, so I'll include some links at the end of the talk.
- The ones you've heard about are probably beautiful and mathematically sound. My colleague Greg Phiel is giving a talk on that kind of thing on Saturday that you should check out. 
- However w/ bindings to ML libraries we are dealing with inherently mutable state and FFI code which doesnt behave nicely.
- In those cases, we need an approach sometimes referred to as the handle pattern or service pattern. 


## ... allows you to
* Encapsulate and hide state inside handles
<!-- .element: class="fragment" -->
* Provide functions to manipulate this state rather than touching it directly
<!-- .element: class="fragment" -->
* Provide interfaces for configuration, creation, and destruction of handles
<!-- .element: class="fragment" -->
Note:
- Replace handle w/ object and function with method and this sounds like Object Oriented Programming.
- You'll see that in practice here it looks quite different. 
- Thats b/c we're essentially managing state outside of Haskell, and using FFI code to constrain input and output
- This method does relies heavily on IO and closely related environments like MonadIO, MonadMask, & MonadResource.
- Whenever you can write things in a pure way, you should attempt to do that and avoid IO. 
- But we've found the handle pattern to be really useful when IO is required


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
  , listVersions :: ModelName -> 
                    IO (Either SourceError [ModelVersion])
}
```
Note:
- This is a handle that represents a model location, either on the local filesystem, S3, a database,etc.
- We have 3 functions.
- We use TFS at work to serve models. 
- TFS expects a model to consist of a directory w/ subdirs for each version. We use this format for other types of ML model as well. 
- Notice that theres no way to delete a model.
- This handle needs to do 2 things behind the scenes: manage scare resources efficiently & maintain version integrity


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
- this is the handle i'm going to spend most of the rest of the talk on, it represents a model that's been loaded into memory.
- Separate training and serving inputs. 
- Used for both supervised learning (cust embeddings) and rl
- Error type is used to drive early stopping behavior w SGD.
- A Monad constraint is typically placed on m by callers. 


#  

```haskell
{-# LANGUAGE TypeFamilies #-}                      -- 1
module Policy.Model.Types where

type ModelHandle' c m =                            
  ModelHandle (LInput c) (SInput c) (Output c) (Error c) m 

class ModelConfig c where
  type LInput c                                    -- 2
  type SInput c
  type Output c
  type Error c
  type Finalizer c

  createModelHandle :: (MonadIO m, MonadMask m)    -- 3
                    => c 
                    -> m (ModelHandle' c m, Finalizer c)
  deleteModelHandle :: (MonadIO m, MonadMask m) 
                    => Finalizer c -> m ()
```

Note: 
- here is the typeclass that we use to manage handles. similar to a factory in OO. 
- its actually a type family, which is a language extension providing a limited form of type-level functions in haskell
- the five types listed there depend on the implementing type c. 
- we'll see two implementations of this class shortly
- we also provide a type alias that basically calls the type level function. i'll be using this everywhere 
- there are two functions: create returns a tuple of the handle itself as well as a destructor
- create model will specialize the type of modelhandle' and finalizer
- here are those constraints i mentioned earlier. the monadmask constraint on the handle provides for async exception handling. all implementations of this interface involve bindings to C or C++ libraries, 


#  

```haskell
{-# LANGUAGE TypeApplications #-}                  
module Policy.Model.API where 

import qualified Control.Exception.Safe as Safe

withModelConfig ::
  forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => c
  -> (ModelHandle' c -> m r)                       -- 1
  -> m r
withModelConfig c act =
  Safe.bracket (createModelHandle c)               
               (deleteModelHandle @c . snd)        -- 2 
               (act . fst)                         -- 3
```
```haskell
bracket :: forall m a b c . MonadMask m =>         
           m a -> (a -> m b) -> (a -> m c) -> m c
```
<!-- .element: class="fragment" -->
Note: 
- heres an example from the api. it consumes an action function that performs some action w the handle
- we feed it a configuration and a handle action, and it creates the handle, performs the action, and deletes the handle in a resource-safe fashion.
- recall that create returns (ModelHandle' c m, Finalizer c)
- the other lang extension we use a lot is called type applications. lets me pass type information explicitly. 
- also note that i'm grabbing the finalizer 
- uses what's referred to as the bracket pattern in haskell. like try / finally or with in python. 
- this is how we run training jobs
- if i talk fast we'll see this in use in the last slide of the talk


