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


# Configuring a Handle

```haskell
{-# LANGUAGE TypeFamilies #-}
module Policy.Source.Types where

class SourceConfig c where
  type Finalizer c
  createSourceHandle :: MonadResource m => c -> m (SourceHandle, Finalizer c)
  deleteSourceHandle :: MonadIO m => Finalizer c -> m ()
```

Note: We generally want to attach richer type information to a handle. Like in this case where the destructor depends on the type of source we have. So we use a type family to capture these dependencies. A type family is a sort of 'type level' function.


# Configuring a Handle

```haskell
module Policy.Source.Local.Config where

newtype LocalSourceConfig = LocalSourceConfig {
    unLocalSourceConfigPath :: FilePath
  } deriving (Eq, Ord, Show)

instance SourceConfig LocalSourceConfig where
  -- no Finalizer, since we create no resources
  type Finalizer LocalSourceConfig = ()

  createSourceHandle c = do
      let sourcePath = unLocalSourceConfigPath c
          handle = SourceHandle {
              loadModel = loadModel' sourcePath
            , saveModel = saveModel' sourcePath
            , listModelVersions = listModelVersions' sourcePath
            }
      pure (handle, ())

  deleteSourceHandle _ = pure ()
```

Note: Here is a trivial implementation using an existing (properly formatted) directory tree.

- now we are in Policy.Source.Local.Config 
- were using a local filepath to configure the handle
- loadModel', saveModel', listModelVersions' are also all specific to this instance and defined in this module or a submodule
- note that the finalizer is trivial b/c there are no resources to clean up (the directory already existed)


# Configuring a Handle

```haskell
module Policy.Source.S3.Config where

import qualified Control.Monad.Trans.Resource as R
import qualified Network.AWS.S3 as S3

data S3SourceConfig  = S3SourceConfig {
    s3SourceConfigAddress :: S3.Address
  , s3SourceConfigEnv :: S3.Env
  }

instance SourceConfig S3SourceConfig where
  type Finalizer S3SourceConfig = R.ReleaseKey

  createSourceHandle c = do
      (tmp, rKey) <- mkTempFolderWithReleaseKey "s3-source-model"
      let handle = SourceHandle {
              loadModel = loadModel' c tmp
            , saveModel = saveModel' c
            , listModelVersions = listModelVersions' c
            }
      pure (handle, rKey)

  deleteSourceHandle = R.release
```

Note: Here is an implementation that abstracts away a remote version of the directory tree stored on S3. 

- we're in a different module Policy.Source.S3.Config 
- we're using an S3 filepath to configure the handle
- the finalizer is non-trivial in this case, since we are copying files to a tmp directory


# Configuring a Handle

```haskell
-- | Create a temp folder and a release key to remove it later
mkTempFolderWithReleaseKey ::
     R.MonadResource m
  => String
  -> m (FilePath, R.ReleaseKey)
mkTempFolderWithReleaseKey template = do
    tmp <- liftIO $ do
        baseTmp <- getCanonicalTemporaryDirectory
        createTempDirectory baseTmp template
    rKey <- R.register $ removeDirectoryRecursive tmp
    pure (tmp, rKey)
```

Note: We use this implementation quite a bit 
- TFS polls production buckets for new versions to load
- Training pipelines write out trained models to buckets
- EMR jobs perform hyperparameter searches and ablation testing
- can either include auth as part of the configuration or pass a token to the handle as an argument


# Using a Handle

```haskell
{-# LANGUAGE TypeApplications #-}
module Policy.Source.API where

import qualified UnliftIO.Exception as Ex

withSourceConfig ::
     forall c m r.
     SourceConfig c
  => MonadResource m
  => MonadUnliftIO m
  => c
  -> (SourceHandle -> m r)
  -> m r
withSourceConfig c act =
    Ex.bracket (createSourceHandle c) (deleteSourceHandle @c . snd) (act . fst)
```

Note: With our SourceConfig typefamily, we are able to define functions using it. Here is the most general one.

- forall proves that this function can be used w/ any sourceconfig that supplies a suitable m.
- 'with*' pattern and callback / cont passing style
- MonadResource and MonadUnliftIO
- brackets
- act . fst
- type application @c


# Using a Handle

```haskell
module Policy.Source.API where

import Policy.Source.Types as ST

loadModel ::
  => MonadIO m
  => SourceHandle
  -> ModelId
  -> Maybe ModelVersion
  -> m (Either SourceError Model)
loadModel h id mv = do
    liftIO $ ST.loadModel h id mv
```

Note:  Can also use the basic 'member functions' directly through a thin API


# Using a Handle

```haskell
module Policy.Source.API where

import Policy.Source.Types as ST

loadModel ::
     HasSourceHandle r SourceHandle
  => MonadReader r m
  => MonadIO m
  => ModelName
  -> Maybe ModelVersion
  -> m (Either SourceError ModelFile)
loadModel n v = do
    h <- view sourceHandle
    liftIO $ ST.loadModel h n v
```

Note:  In practice we place an additional MonadReader constraint on the handle to parametrize over the handle. This is b/c we use sources in a variety of situations. 


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


