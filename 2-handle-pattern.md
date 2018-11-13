# The Handle Pattern

* Encapsulating and hiding state inside objects
<!-- .element: class="fragment" -->
* Providing methods to manipulate this state rather than touching it directly
<!-- .element: class="fragment" -->
* Providing interfaces for configuration, creation, and destruction of objects
<!-- .element: class="fragment" -->

Note:

Haskell design patterns are sortof a blogosphere phenom, so I'll include some links at the end of the talk.

In Haskell, we try to capture ideas in beautiful, pure and mathematically sound patterns, for example Monoids. But at other times, we can’t do that. We might be dealing with some inherently mutable state, or we are simply dealing with external code which doesn’t behave nicely.


# Sounds Like OO?

https://softwareengineering.stackexchange.com/questions/46592/so-what-did-alan-kay-really-mean-by-the-term-object-oriented
<!-- .element: class="fragment" -->

Note:

In those cases, we need another approach. What we’re going to describe feels a bit similar to Object Oriented Programming.

As you can see, it is not exactly the same as Alan Kay’s original definition of OOP, but it is far from the horrible incidents that permeate our field such as UML, abstract factory factories and broken subtyping.


# A Handle

```haskell
data DatabaseHandle = DatabaseHandle {
     hPool   :: Pool Postgres.Connection
   , hCache  :: IORef (PSQueue Int Text User)
   , hLogger :: Logger.Handle  -- Another handle!
   , …
}
```

Note: It's also possible to do more involved operations involving other handles, MVars, IORefs, TVars, Chans etc. We refrain from doing this.


# A Simple Handle

```haskell
module Policy.Source.Types where

data SourceHandle = SourceHandle {
     loadModel :: ModelName -> Maybe ModelVersion -> IO (Either SourceError ModelFile)
    -- ^ Load a model file onto the local file system. If no version is provided get the latest version.
   , saveModel :: ModelName -> ModelFile -> IO (Either SourceError ModelVersion)
    -- ^ Save a model file, bumping the latest version.
   , listModelVersions :: ModelName -> IO (Either SourceError [ModelVersion])
    -- ^ List all available versions for a given model.
}

newtype ModelFile = ModelFile { unModelFile :: FilePath } 
```

Note:

We use TFS at work to serve a wide variety of models. TFS expects a model to consist of a directory w/ subdirs for each version. So we tend to use this format everywhere. 

the purpose of this handle is to broker safe access to the underlying dir. by safe i mean that it maintains certain properties (ie if a model has n versions and you call saveModel it will now have n+1 versions)

How is this sim/diff from a type class?


# Parametrized Handles

```haskell
module Policy.Model.Types where

data ModelHandle i j o e m = ModelHandle {
     learn :: i -> m ()
   , score :: j -> m o
   , error :: i -> m e 
}
```

Note:
A MonadIO constraint is typically placed on m by callers.

- Separate training and serving inputs. 
- Used for both supervised learning (cust embeddings) and rl
- Error type is used to drive early stopping behavior w SGD.

Since there are so many type parameters configuration


#  Parametrized Handles

```haskell
{-# LANGUAGE TypeFamilies     #-}
{-# LANGUAGE TypeApplications #-}
module Policy.Model.Types where

import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Exception.Safe (MonadMask)
import Policy.Source.Types as PT

type ModelHandle' c m =  ModelHandle (TrainingInput c) (ServingInput c) (Output c) (Error c) m

class ModelConfig c where
    type TrainingInput c
    type ServingInput c
    type Output c
    type Error c
    type Finalizer c

    createModelHandle :: (MonadIO m, MonadMask m) => c -> m (ModelHandle' c m, Finalizer c)
    deleteModelHandle :: (MonadIO m, MonadMask m) => Finalizer c -> m ()
    ...
```

Note: 

- here are those monadIO constraints i mentioned earlier
- several implementations of this interface involve bindings to C or C++ libraries, 
- the monadmask constraint on the handle provides for async exception handling. I'll walk through one example of this later.
- we provide a simple type alias to hide the dependant typing


# Parametrized Handles

```haskell
module Policy.Model.API where 

import qualified Control.Exception.Safe as Safe

withModelConfig ::
     forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => c
  -> (ModelHandle' c -> m r)
  -> m r
withModelConfig c act =
  Safe.bracket (createModelHandle c) (deleteModelHandle @c . snd) (act . fst)
```

Note: With most of the complexity hidden in the modelconfig type family, our API becomes very clean & simple like before.


