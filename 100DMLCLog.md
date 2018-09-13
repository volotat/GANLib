# 100 Days Of ML - LOG

## Day 1 : July 7, 2018
 
**Today's progress** : Plotter updated, so now it plots all available history labels. Also diving into Progressive GAN so I can start to implement it as a next one.

**Link to work:**   [Commit](https://github.com/Mylittlerapture/GANLib/commit/6b58af27fcd55b3b32efe17219e3cc952a2df2b4)


## Day 2 : July 8, 2018
 
**Today's progress** : Start working on Progressive GAN implementation. For now I'm going to implement basic idea of growing layers. Other ideas from original paper will be added gradually. 

**Link to work:**   [Commit](https://github.com/Mylittlerapture/GANLib/commit/eb270d9d9f20295f13cefb0f1251ecad30709a2c), [Commit](https://github.com/Mylittlerapture/GANLib/commit/85c44053e3bf3dff5381dd8938318bdd04ea35cf)


## Day 3 : July 9, 2018
 
**Today's progress** : Progressive GAN is now growing during training. Not sure about overall structure though. Probably will change it later. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/54816d45c9b02e308712e765e78dee97b55b4b56)


## Day 4 : July 10, 2018
 
**Today's progress** : Added template for Adversarial Autoencoder with little twist to it, trying to test some ideas. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/77e18eacb2447458bc809a1ac4bf0a733ae05a0f)


## Day 5 : July 11, 2018
 
**Today's progress** : Today I experimented with domain to domain encoding, with a little bit of success. Because result code end up way different from AAE I split it to different class called DAAE (Domain based Adversarial Autoencoder)

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/eadf7819fd1e309c70e461b30b808c4cf640fc28)


## Day 6 : July 12, 2018
 
**Today's progress** : Cleaned up DAAE code and eventually improved results a lot by changing a way how discriminator work.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/91139f51cb4fbe45e122df24aa92ea9dc6c394fb)


## Day 7 : July 13, 2018
 
**Today's progress** : Turns out my DAAE works very similar to DiscoGAN so I renamed it. It a bit sad that this idea far away from new, on the other hand it is a good reminder to spend more time on research. Also made small description for some GANs within source code.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/eb2b06198b3cccde8bd7b7f1625850dc90cf6842), [Commit](https://github.com/Mylittlerapture/GANLib/commit/758ab2e30811ddebf7d2bc223dd2f89e9a135b62)


## Day 8 : July 14, 2018
 
**Today's progress** : AAE and AAE_test code changed. Unfortunately it's still not working as it supposed to, at least not good enough. Also added descriptions for remaining GANs.  

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/35d9d23250f101b9fb3902b14297f36c3809d309)


## Day 9 : July 15, 2018
 
**Today's progress** : AAE now working correctly and store some of training history.  

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/a4839920a56b689ad8c6e7864b0ec00c30f7b6b2)


## Day 10 : July 16, 2018
 
**Today's progress** : Changed a way how stable mode works for GAN and CGAN. Now it actually produce much better results for GAN and roughly the same for CGAN but faster.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5d4a42bb09972159ddc2ff5bf45664d7752f174a)


## Day 11 : July 17, 2018
 
**Today's progress** : Update stable mode for Progressive GAN, models moved out of class.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5ae9a42c6937b834af1ac97be0cac1c710bdb0d3), [Commit](https://github.com/Mylittlerapture/GANLib/commit/4c30d088003219506f1dbea7cd778df77a3e623a)


## Day 12 : July 18, 2018
 
**Today's progress** : Fixed savings and verbosity mode for every module. Small improvements in ProgGAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/52bc82b91e3232e5e85e6c4cdf07649bd8301e15)


## Day 13 : July 19, 2018
 
**Today's progress** : Changed a way of storing weights and fix metric test in ProgGAN. Added Pixelwise normalization layer to utils and way of using it to ProgGAN_test.py 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/818bdb9c8a76dd6f4995d043622652a0eba55d25)


## Day 14 : July 20, 2018
 
**Today's progress** :  Metric test in ProgGAN fixed again. Added MiniBatchStddev layer from ProgGAN paper.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/1e4345c1b6d4b0f4f984943c1ca51651bfeeb601)


## Day 15 : July 21, 2018
 
**Today's progress** :  Switch MiniBatchStddev code to tensorflow version. Added MiniBatchDiscrimination layer to utils.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/50dadc9f5a851beaa13479fc84db4d518521faf9)


## Day 16 : July 22, 2018
 
**Today's progress** :  Made a draft version of smooth fading between layers in ProgGAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/60782822e1fb326041acb9640f5ba7bcee86820f)


## Day 17 : July 23, 2018
 
**Today's progress** :  Today I spent time testing ProgGAN on high resolution faces dataset.


## Day 18 : July 24, 2018
 
**Today's progress** :  Code for MinibatchDiscrimination layer completely changed. Some improvements in ProgGAN. Still testing it on bigger dataset, but results not so good yet. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/a1a7b1cb77616f1364d7ff9bf4051bbc0440cb47)


## Day 19 : July 25, 2018
 
**Today's progress** :  Added rough implementation of improved Wasserstein loss into ProgGAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5bd92f293c7379c1121908e6e1877f8f98f018c2)


## Day 20 : July 26, 2018
 
**Today's progress** :  Some improvements in ProgGAN. Now it produce more consistent results, but still breaks if model become too big.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/6de25995c0b36f7d65bde13a7dd7a2c8d3e2991f)


## Day 21 : July 27, 2018
 
**Today's progress** :  Still working on improvements for ProgGAN. For now there are realized most of significant features from the paper.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/f114b3adb330787c19e8431d6a957ce2339d99a9)


## Day 22 : July 28, 2018
 
**Today's progress** :  After a number of experiments ProgGAN seems to work a bit better, although far from perfect. I guess it is best result that I can get by now.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/05307d588ecc1341b14bb796df6737c8bdb32c78)


## Day 23 : July 29, 2018
 
**Today's progress** :  Some changes in ProgGAN code to make it more readable and clear.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/e04d2e005b734b5ce10c09210f449de91e860f51)


## Day 24 : July 30, 2018
 
**Today's progress** :  Complete working on ProgGAN for now.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/d2a0fcb96b15369cae8990383bd07a34cbbe57fe)


## Day 25 : July 31, 2018
 
**Today's progress** :  Made a template for CramerGAN on which I'm going to work next.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/d7c17ff5e83e495397e0e8bda5d614d15c4663b0)


## Day 26 : August 1, 2018
 
**Today's progress** :  A bit of work on CramerGAN, just building the understanding of how it works

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/6e8ff2d5282a917840798ad5b1fea8c7c2577e62)


## Day 27 : August 2, 2018
 
**Today's progress** :  Building CramerGAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/b10865ceda5cbea0b08388a79e34f32ab753be77)


## Day 28 : August 3, 2018
 
**Today's progress** :  Running first tests on CramerGAN and it is kind of works. What is very surprising, because it far from complete.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/7a384068efe5dc2353b63a797a9ba259c1debab1)


## Day 29 : August 4, 2018
 
**Today's progress** :  Most important code for CramerGAN is now implemented. There are some tensorflow code for now, I will rewrite it as Keras backend later.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/c0a5e0bf3f213d33ad53b07a8911210ee5e8bd7e)


## Day 30 : August 5, 2018
 
**Today's progress** :  CramerGAN almost completed, just few minor things left. It seems to work fine. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/d74e2735e3e723e8867dc4241f069e900be1b13c)


## Day 31 : August 6, 2018
 
**Today's progress** :  CramerGAN is now completed. All tensorflow code replaced with Keras backend.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/e24d4594f303b6ab32b2406af9c59c0d5f5aa037)


## Day 32 : August 7, 2018
 
**Today's progress** :  Start to rewrite library in a way that all GANs would be child classes of original GAN. This would reduce amount of unnecessary code.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/c6c1925920b4829477bd28f381003c345e8f8f1f)


## Day 33 : August 8, 2018
 
**Today's progress** :  Continue rewriting library in a better way.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/89c4da2762eea202fab70f3ab579bec5d90ed660)


## Day 34 : August 9, 2018
 
**Today's progress** :  GAN, CramerGAN, CGAN are now rewrited. From now Progressive GAN would be just a special case of improved Wasserstein GAN, that not implemented yet.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/f8fd97fa27948ed894c4eb3dbd2d210704184da2)


## Day 35 : August 10, 2018
 
**Today's progress** :  Added example of simplest possible GAN to the library.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/02c72d33f8b47e9f55d965f4ab5119bd22ac6a07)


## Day 36 : August 11, 2018
 
**Today's progress** :  Changed the way of saving and loading models.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/9b6a59eac6eab20eb50b2cb5473ce5b990419d3c)


## Day 37 : August 12, 2018
 
**Today's progress** :  Rebuilding Adversarial Autoencoder and trying to make it work properly.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/b9321eb56feca7bc2721cf3f68a59afdfa531ec0)


## Day 38 : August 13, 2018
 
**Today's progress** :  Complete Adversarial Autoencoder. Realize that there is necessity to change a way of history processing.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/8691f04f3cf9d0964aa22c9a1475b9b11cc40247)


## Day 39 : August 14, 2018
 
**Today's progress** :  Start building Improved version of Wasserstein GAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/b59dbc8b6254fd2a436e0511c07a5da3afcab0e5)


## Day 40 : August 15, 2018
 
**Today's progress** :  Created graph for Improved Wasserstein GAN. Module was renamed to WGAP_GP because gradient penalty is most important part of this algorithm.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/a86eb2bea917b2c43a22002005fb069585438258)


## Day 41 : August 16, 2018
 
**Today's progress** :  Got first good results with WGAP GP. Now it's possible to rewrite Progressive Growing of GANs in a proper way. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5adc595bb01fee18789219b0c540f02cef14a5ac)


## Day 42 : August 17, 2018
 
**Today's progress** :  Started reimplementing Progressive Growing of GANs as example of WGAN GP use case. 

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/48e4c4bec7f3f870d53636eb3f921b996f7b616b)


## Day 43 : August 18, 2018
 
**Today's progress** :  Continue reimplementing Progressive Growing of GANs in a better way. A bit messy right now, but will fix it later.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/3aa15f97740e725115b76509c58f613ac0bdee3c)



## Day 44 : August 19, 2018
 
**Today's progress** :  Still rebuilding PG GAN while also improving overall structure of the library.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/629ba2d19e60930e4b19c040a24208b88c20e46a)


## Day 45 : August 20, 2018
 
**Today's progress** :  PG GAN almost complied. API of the library a bit more intuitive now.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/0c4831816145f65eb818fe3f1baf55e019e6a2ae)


## Day 46 : August 21, 2018
 
**Today's progress** :  Completed PG GAN example.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/ea7296dbbf451ce05b10103cf0c5f94e03db6d10)


## Day 47 : August 22, 2018
 
**Today's progress** :  Started rebuilding DiscoGAN. Also experimenting with some ideas for image generation.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/f7b9a676827f0d9a9e0b09d1d25f2af2208ebccb)


## Day 48 : August 23, 2018
 
**Today's progress** :  Continue working on DiscoGAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5090320d953eead897f188b25bad9abc1deb61ba)


## Day 49 : August 24, 2018
 
**Today's progress** :  Completed with DiscoGAN, but it still do not store history. I need to rebuild it first.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/4ae7626b18bb71421f6e0ff07ac40cc82a5d168e)


## Day 50 : August 25, 2018
 
**Today's progress** :  Trying to improve history collection system.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/81082a53663b108879a282e5bbe5fc59ec532533)


## Day 51 : August 26, 2018
 
**Today's progress** :  History collection mechanism significantly improved. Now it much more general and easier to manage.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/965b489064de976b444042b55b74f100aaba3a4d)


## Day 52 : August 27, 2018
 
**Today's progress** :  Started implementing inception score as new metric.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/a86ce534ddc948c99ab2569b1bc6bc5cd15ce266)


## Day 53 : August 28, 2018
 
**Today's progress** :  Inception score is now implemented and it looks like it works reasonably well.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/d46949b851493db987fdf857be5480a4ce733385)


## Day 54 : August 29, 2018
 
**Today's progress** :  Moving forward I decide to build Tensorflow implementation for each GAN meanwhile trying to remain API of the library.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/963bb9c1c8d8d490f46d55ab423b1c66fe87614e)


## Day 55 : August 30, 2018
 
**Today's progress** :  Made Tensorflow implementation of GAN work. I will probably switch whole library to Tensorflow in the future as it seems like much more suitable for GAN related tasks.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/32c02ec39fca1d237293ca216dcd0a4049f057cd)


## Day 56 : August 31, 2018
 
**Today's progress** :  Some experiments with Tensorflow version of GAN.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/98c503e22efc388a34fb9d2adf758d4c540473e2)


## Day 57 : September 1, 2018
 
**Today's progress** :  Finally achieve stability with Tensorflow version of GAN. There is still some work need to be done with this module though.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/d7d16710f68384fb0ea019266c74e5ce1cbe51f0)


## Day 58 : September 2, 2018
 
**Today's progress** :  Tensorflow version of GAN collect history now. For some reason it performs a bit worse that Keras implementation but trains faster.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/23e0431a8fe5072ed719f826c12626c2bb56bdbc)


## Day 59 : September 3, 2018
 
**Today's progress** :  Small improvements in PG GAN example. Tensorflow version of GAN deleted from main branch and moved to new one.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/aa5a1d8fb9daf432885478801ca5a727900c5d84)


## Day 60 : September 4, 2018
 
**Today's progress** :  All optimization distances are now options on GAN initiation stage. So we don't have to have many different modules for different types of GANs if main structure is the same.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/5daa0753c5cb9e694c6c081695f820415b3b6257)


## Day 61 : September 5, 2018
 
**Today's progress** :  Distances now rewritten as classes and only cramer distance left to be realized.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/e4b4a9a35112817fd55e41e7e2e6ae3520cedd3e)


## Day 62 : September 6, 2018
 
**Today's progress** :  Added new version of Cramer distance to distances calss.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/9132ef824f0a933a2ede503c2920604b3643a85b)


## Day 63 : September 7, 2018
 
**Today's progress** :  Started to build conditional GAN in tensorflow branch.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/cf4e33cbed4497e3df2074bdfd217ef41b33d097)


## Day 64 : September 8, 2018
 
**Today's progress** :  Rewrote gan example for tensorflow base, also improved conditional gan so it's working at least for minmax distance.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/65c21c55ef1479cb69a7391444f12bc137bb7271)


## Day 65 : September 9, 2018
 
**Today's progress** :  Trying to merge conditional gan with gradient penalty, no luck yet.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/ae1712687f300c7d2831b3e06fc3d09ccdacd658)


## Day 66 : September 10, 2018
 
**Today's progress** :  Conditional gan now storing history and working with almost all distances except Cramer.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/67000400e25081bd1336f3014858560fc632b2d2)


## Day 67 : September 11, 2018
 
**Today's progress** :  Tried to fix bug within Cramer distance.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/35e3c79536ab1ca7dbce1cb93dd07e071aa271e4)


## Day 68 : September 12, 2018
 
**Today's progress** :  Cramer distance bug fixed. Conditional GAN now supports all available distances.

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/4af1da77d7bdaccc4a44a913b1411c1ec6af5839)


## Day 69 : September 13, 2018
 
**Today's progress** :  Started adding DiscoGAN to Tensorflow branch

**Link to work:**  [Commit](https://github.com/Mylittlerapture/GANLib/commit/6bdeda19f704c05f0d4b564f6900bad21fed9d93)
