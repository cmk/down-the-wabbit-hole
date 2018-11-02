# The Handle Pattern

Note:

Haskell design patterns are sortof a blogosphere phenom, so I'll include some links at the end of the talk.

In Haskell, we try to capture ideas in beautiful, pure and mathematically sound patterns, for example Monoids. But at other times, we can’t do that. We might be dealing with some inherently mutable state, or we are simply dealing with external code which doesn’t behave nicely.

In those cases, we need another approach. What we’re going to describe feels suspiciously similar to Object Oriented Programming:

Encapsulating and hiding state inside objects
Providing methods to manipulate this state rather than touching it directly
Coupling these objects together with methods that modify their state
As you can see, it is not exactly the same as Alan Kay’s original definition of OOP4, but it is far from the horrible incidents that permeate our field such as UML, abstract factory factories and broken subtyping.

Pretty much any sort of Haskell code can be written in this particular way, but that doesn’t mean that you should. This method relies heavily on IO. Whenever you can write things in a pure way, you should attempt to do that and avoid IO. This pattern is only useful when IO is required

https://www.schoolofhaskell.com/user/meiersi/the-service-pattern
https://jaspervdj.be/posts/2018-03-08-handle-pattern.html
https://www.tweag.io/posts/2018-04-25-funflow.html

https://hackernoon.com/the-has-type-class-pattern-ca12adab70ae
http://www.parsonsmatt.org/2018/03/22/three_layer_haskell_cake.html
http://www.parsonsmatt.org/2018/04/10/transforming_transformers.html --opposite approach
https://www.fpcomplete.com/blog/2017/06/readert-design-pattern


# A Simple Handle

```haskell
module Model.Source.Types where

data SourceHandle = SourceHandle {
    loadModel :: ModelId -> Maybe ModelVersion -> IO (Either LoadError ModelFile)
    -- ^ Load a model file onto the local file system. If no version is provided get the latest version.
  , saveModel :: ModelName -> ModelFile -> IO (Either SaveError ModelVersion)
    -- ^ Save a model file, bumping the latest version.
  , listVersions :: ModelId -> IO ([ModelVersion])
    -- ^ List all available versions for a given model.
  }
newtype ModelFile = ModelFile { unModelFile :: FilePath } 
```

Note:

We use TFS at work to serve a wide variety of models. TFS expects a model to consist of a directory w/ subdirs for each version. So we tend to use this format everywhere. the purpose of this handle is to broker safe access to the underlying dir. by safe i mean that it maintains certain properties (ie if a model has n versions and you call saveModel it will now have n+1 versions)
How is this sim/diff from a type class?

The internals of the Handle typically consist of static fields and other handles, MVars, IORefs, TVars, Chans… With our Handle defined, we are able to define functions using it. 

It's also possible to do more involved operations there (pools, io refs, other handles, etc)
data DatabaseHandle = DatabaseHandle
    { hPool   :: Pool Postgres.Connection
    , hCache  :: IORef (PSQueue Int Text User)
    , hLogger :: Logger.Handle  -- Another handle!
    , …
    }


# Using a Handle

```haskell
module Model.Source.API where
import Model.Source.Types as ST

loadModel ::
  => MonadIO m
  => SourceHandle
  -> ModelId
  -> Maybe ModelVersion
  -> m (Either LoadError Model)
loadModel h id mv = do
  liftIO $ ST.loadModel h id mv
```

Note: Discuss MonadReader?


# Dependently Typed Handles

```haskell
{-# LANGUAGE GADTs      #-}
{-# LANGUAGE RankNTypes #-}

data ModelHandle ti si o e where
  ModelHandle :: {
      learn :: MonadIO m => ti -> m ()
    , score :: MonadIO m => si -> m o
    , error :: MonadIO m => ti -> m e
  } -> ModelHandle ti si o e
```

Note:
I want to use this to train models in some map-reduce framework, serve models through some model serving framework, etc.
Use GADTs to avoid exposing the monadic parameter m.
Separate training and serving inputs. Error type is used to drive SGD behavior.


#  Dependently Typed Handles

```haskell
{-# LANGUAGE TypeFamilies     #-}
{-# LANGUAGE TypeApplications #-}

import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Exception.Safe (MonadMask)
import qualified Data.Tagged as T

class ModelConfig c where
    type ServingInput c
    type TrainingInput c
    type Output c
    type Error c
    type Finalizer c

    readModelConfig :: (MonadIO m) => PT.ModelFile -> m c

    createModelHandle :: (MonadIO m, MonadMask m) => c -> m (ModelHandle' c, Finalizer c)

    createModelHandle' :: (MonadIO m, MonadMask m) => T.Tagged c ModelFile -> m (ModelHandle' c, Finalizer c)
    createModelHandle' mf = do
      c <- liftIO $ readModelConfig (T.untag mf)
      createModelHandle @c c

    deleteModelHandle :: (MonadIO m, MonadMask m) => Finalizer c -> m ()

type ModelHandle' c =  ModelHandle (TrainingInput c) (ServingInput c) (Output c) (Error c)
```

Note: Sometimes we want to attach richer type information to a handle. Like in this case where the input, output, error, and destructor types depend on the kind of model we're serving. So we use a type family to capture these dependencies. A type family is a sort of 'type level' function.
This is probably the most complex slide in this talk, sorry.


# Model Config API


```haskell
import qualified Control.Exception.Safe as Safe

withModelConfig ::
     forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => c
  -> (ModelHandle' c -> m r)
  -> m r
withModelConfig c act =
  Safe.bracket (createModelHandle c) (deleteModelHandle @c . snd) (act . fst)

withModelFile ::
     forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => ModelFile
  -> (ModelHandle' c -> m r)
  -> m r
withModelFile mf act =
  Safe.bracket
    (createModelHandle' (T.Tagged mf :: T.Tagged c ModelFile))
    (deleteModelHandle @c . snd)
    (act . fst)
```

Note: With most of the complexity hidden in the modelconfig type family, our API becomes very clean & simple like before.


# Model Config API

```haskell
withModelFile ::
     forall c m r. (ModelConfig c, MonadIO m, MonadMask m)
  => ModelFile
  -> (ModelHandle' c -> m r)
  -> m r
withModelFile mf act =
  Safe.bracket
    (createModelHandle' (T.Tagged mf :: T.Tagged c ModelFile))
    (deleteModelHandle @c . snd)
    (act . fst)
```

Note: With most of the complexity hidden in the modelconfig type family, our API becomes very clean & simple like before.


# A TF Handle

```haskell
import qualified TensorFlow.Core as TF
import qualified Data.Vector as V
import qualified Platform.Model.Types as Types

type Label = (TF.TensorData Int32, TF.TensorData Int32)

type Embedding = Types.ModelHandle (Label,Label) (V.Vector Float)

data Config = Config { populationSize :: Word32, embeddingDim :: Word32 }

-- createHandle :: (MonadIO m, MonadMask m) => c -> m (ModelHandle (Input c) (Output c), Finalizer c)
-- TF.runSessionWithOptions :: (MonadIO m, MonadMask m) => TF.Options -> TF.SessionT m a -> m a
createHandle :: (MonadIO m, MonadMask m) => TF.Options -> Config -> m Embedding
createHandle opts config = TF.runSessionWithOptions opts $ TF.build (createEmbedding config)

createEmbedding :: Config -> TF.Build Embedding
createEmbedding config = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
        populationSize' = populationSize config
        embeddingDim' = embeddingDim config

    customers <- TF.placeholder [batchSize, populationSize']
    let custVecs = TF.oneHot customers (fromIntegral populationSize') 1 0

    w1_init <- randomParam embeddingDim' [populationSize', embeddingDim']
    w1 <- TF.initializedVariable w1_init
    let embeddings = (custVecs `TF.matMul` TF.readValue w1) 

    w2_init <- randomParam embeddingDim' [embeddingDim', populationSize']
    w2 <- TF.initializedVariable w2_init
    b2 <- TF.zeroInitializedVariable [populationSize']
    let scores = (embeddings `TF.matMul` TF.readValue w2) `TF.add` TF.readValue b2

    predictions <- TF.render @TF.Build @LabelType $
                   TF.argMax (TF.softmax scores) (TF.scalar (1 :: LabelType))

    -- Create training action.
    labels <- TF.placeholder [batchSize]
    let labelVecs = TF.oneHot labels (fromIntegral populationSize') 1 0
        loss = TF.reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits scores labelVecs
        params = [w1, w2, b2] 

    trainStep <- TF.minimizeWith TF.adam loss params

    lossTensor <- TF.render loss
    output <- TF.render embeddings
    return Embedding  {
          learn (inputs, outputs) = TF.runWithFeeds_ [
                TF.feed customers inputs
              , TF.feed labels outputs
              ] trainStep
        , predict (inputs, _)= TF.runWithFeeds [TF.feed customers inputs] $ TF.readValue w1
        , loss (inputs, outputs) = TF.unScalar <$> TF.runWithFeeds [
                TF.feed customers inputs
              , TF.feed labels outputs
              ] lossTensor
        }
```

Note: that I'm simply creating the TF graph here. this requires no data dependencies


# A VW Handle


