<h1><b>Wine recommender</b></h1>

Goal: Build a model that help a wine lover find a wine similar to what she/he likes, and at a price point she/he desires.

<h2>BUSINESS MOTIVATION</h2>

A wine lover often wants to find a wine…

1. with a certain set of characteristics. For example, “_A refreshing sparkling rose with orange and peach” or “A robust red with cinnamon, coconut, and nutmeg.”_
2. that has been positively reviewed, but perhaps it was not available at a nearby store or even on-line, so an alternative is desired, or the reviewed wine was too expensive, so a lower-priced one is sought.

A different approach is needed to satisfy these two business needs. Whereas in building the price model, we ignore the text by the wine reviewers, here will be completely lean into them. Here the model not only needs to know the words such as _orange/citrus_ and _strong/robust_ are similar in meaning, but also the overall “sentiment” of the text.

<h2>EXAMPLES OF THE WINE RECOMMENDER</h2>

The best way to demonstrate how well the model works is by examples.  Below shows the response to the input of text of the user's desired wine and price.

Your input text = “<b>I want a bubbly fruity refreshing sparkling wine</b>." Your price = $30

<figure>
    <img src='./images/winerec_bubbly_fruity_refreshing_sparkling.png' width="800">
    <figcaption>Figure: "I want a bubbly fruity refreshing sparkling wine." Your price = $30</figcaption>
</figure><br></br>

Your input text = <b>Robust red cherry with vanilla cinnamon coconut and nutmeg</b>”. Your price = $60
<figure>
    <img src='./images/winerec_robust_red_cherry_vanilla_cinnamon_coconut_nutmeg.png' width="800">
    <figcaption>Figure: "Robust red cherry with vanilla cinnamon coconut and nutmeg" Your price = $60</figcaption>
</figure><br></br>

Your input text = "<b>a mature creamy chardonnay aged with oak</b>". Your_price = $35
<figure>
    <img src='./images/winerec_mature_creamy_chardonnay_aged_with_oak.png' width="800">
    <figcaption>Figure: "A mature creamy chardonnay aged with oak" Your price = $35</figcaption>
</figure><br></br>

<b>Note: More wine recommender examples can be found in the [wine_recommender/images/](<./images/>) folder.</b>

As one can see, the model works very well. It can be used to help wine lovers look for the wine at they want at a price point they can afford.

<h2>HOW WE DETERMINE <i>WINE SIMILARITY</i></h2>

The model above is based on calculating the similarity between the descriptions and prices of the input and each of the wine in the data base.  That is, the overall similarity is product of the description and price similarities.  Mathematically, it is represented as follows:

<i><h4><p style="text-align:center;">Similarity<sub>total</sub> = Similarity<sub>text</sub>  x  Similarity<sub>price</sub><p></h4></i>

<u><h3><i>Similarity<sub>text</sub>: Using Sentence Transformers</i></h3></u>

We want to determine similarity by the actual intrinsic characteristics of the wine as interpreted by that is less biased.  Thus we eschew the data we used in the wine-price model since that model is trained on words derived by someone in the marketing department.  Instead we will fully leverage the taster’s reviews (“description”). Ideally we would have to compress the info with an unsupervised learning model. Training this kind of models requires a very large data set, much larger than the relatively measly 110k in this data set. In the past, we would have to train the model ourselves, using a framework such as WORD2VEC, and fine-tune them to our specific needs. Luckily today there are publicly available models that can accomplish this task. The approach to training this kind of “sentence transformers” involves creating a vector space (embeddings) that map query-response pairs as close to each other into multi-dimensional vector space (See [semantic search](<https://www.sbert.net/examples/applications/semantic-search/README.html>)). A large number of these public sentence transforms can be found in at [hugging face](<https://huggingface.co/sentence-transformers>).

Each of these transformers converts a text to an embedding vector of 768 dimensions. There are many to choose from. We picked the [msmarco-distilbert-dot-v5 transformer](<https://github.com/microsoft/msmarco/blob/095515e8e28b756a62fcca7fcf1d8b3d9fbb96a9/Datasets.md>) (from Microsoft), which is based on the ranked results of the corpus is 3.2 million documents and 367,013 queries, with the assumption that a document that produced a relevant passage is usually a relevant document . An important reason we selected this transformer is it can accommodate the longest text (512 words). While the longest review text in the present corpus is only 135 words long (after removing stop words), it is possible in the future a review can have an outlying length of > 256 words, which is the transformer with the next longest word limit.

<h3>Why <i>Cosine Similarity</i> was chosen over <i>Euclidean Distance.</i></h3>

Once the review texts have been transformed to embeddings, the similarity between any two wines can be estimated by calculating the Cosine Similarity (CS) or the Euclidean Distance (ED) between the two embeddings. While it is difficult to quantify the relative effectiveness of two approaches, it is possible to see cosine similarity leads to better results.

For example, for the input description “**_I want a bubbly fruity refreshing sparkling wine_**.”

Here are the top 10 results produced by **cosine-similarity (Excellent)**.

<figure>
    <img src='./images/winerec_cosine_similarity.png' width="800">
    <figcaption>Figure: Most similar wine to <i>I want a bubbly fruity refreshing sparkling wine</i> by <b>Cosine Similarity</b>.  Note the results are very good.</figcaption>
</figure>
As one can see, they are excellent responses to the input text.

Here are the top 10 results from **Euclidean distance (bad)**.

<figure>
    <img src='./images/winerec_euclidean_dist.png' width="800">
    <figcaption>Figure: Most similar wine to <i>I want a bubbly fruity refreshing sparkling wine</i> by <b>Euclidean Distance</b>.  Note the results are very bad.</figcaption>
</figure>

These results are simply bad. None of the wines are even sparkling wines.  But why???

<h4>Noise</h4>

Why did CS produce so much better results? One likely explanation is the notorious [“curse of dimensionality”](<https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity>). The embeddings have a dimension of 768. At such large dimensions, points in general are very far away from each other. The relative distance between any two points is thus very sensitive to noise in the system. CS on the other hand, depends only on the dot product between two vectors (their lengths do not matter), so it is less sensitive to noise.

<h4>Generic wines have shorter Euclidean lengths.  In other words, they are closer to the center of the embedding space, so they are closer to more wines and are recommended more often.</h4>

Further investigation revealed another explanation: The most similar wines by ED tend to have the shorter Euclidean vector length. The same wine with short Euclidean lengths to the center keep appearing regardless of our input. The likely explanation is these are wines that exist nearer to the center of the vector space, so their descriptions tend to be more generic, and there are _closest_ most other wines. In our examples, these recommended wines generated by ED are so generic that they don’t even resemble the specific wine they are supposed to be similar too. On the other hand, the similar wines generated by CS are on the edges of the vector spaces, so farther from the center, meaning they are more specific. This is probably why they better tailored to the text input.

<figure>
    <img src='./images/winerec_hist_ed.png' width="800">
    <figcaption>Figure: Histogram of embed vector lengths</figcaption>
</figure>

<h3>PRICE SIMILARITY</h3>

No wine consumer will ignore the price of a wine in the selection process. Thus any good wine recommender must account for it.  We employ the following function, which is the normalized variance of the prices, multiplied by a _price importance_ (0 ≤ price_imp ≤ 1) factor.

<h4><p style="text-align:center;"><i>Similarity<sub>price</sub> = exp<sup> -(price<sub>interest</sub> - price <sub>another wine</sub>) <sup>2 </sup>price_importance <sup>2 </sup>/price<sub>interest</sub><sup>2</sup></i></p></h4>

The only parameter is <u>price_importance</u>.  Its effects can be seen in the figure.  A larger <i>price_importance</i> make the <i>Similarity<sub>price</sub></i> effect stronger, and a smaller <i>price_importance</i> makes it weaker.

Note in our representation, the impact of the <i>price_importance</i> scales exponentially. The figure below shows its impact. A factor of 1 would be too powerful and it would narrow the choices too much. We choose a factor 0.01 (green). However, this can be tailored by the user as she/he sees fit.


<figure>
    <img src='./images/winerec_priceimp.png' width="800">
    <figcaption>Figure: Price importance factor</figcaption>
</figure>
See notebooks in the folder [link.](./notebooks/)

1.  <b><i>create_description_embeddings.ipynb</i></b> for creating the description embeddings.

2.  <b><i>fine_tune_similarity.ipynb</i></b> for fine tuning the various parameters.

3.  <b><i>find_most_similar_wines.ipynb</i></b> for demonstrating the modeling performing wine recommendation based on the user's input.