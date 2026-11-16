/*
 * FAQ chatbot engine for the browser — a JavaScript copy of chatbot.py.
 *
 * Netlify only hosts static files, so the live demo can't run Python. export_web.py
 * exports the trained model (TF-IDF vocabularies, IDF weights, FAQ vectors, NLTK
 * stopwords and lemmas) to model.json, and this file repeats the same steps:
 *   preprocess -> TF-IDF (word + character n-grams) -> cosine similarity -> best FAQ.
 */
(function (global) {
  "use strict";

  let model = null;
  let stopwords = null;

  // ---------------------------------------------------------------- text processing
  function normalize(text) {
    // same as nlp_utils.normalize: lowercase, drop punctuation, collapse spaces
    return text
      .toLowerCase()
      .replace(/’/g, "'")
      .replace(/[^\p{L}\p{N}_\s']/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function splitContraction(token) {
    // NLTK's tokenizer splits "don't" -> "do" + "n't", "what's" -> "what" + "'s", "cannot" -> "can" + "not"
    if (token === "can't") return ["ca"];
    if (token === "won't") return ["wo"];
    if (token === "cannot") return ["can", "not"];
    if (token.endsWith("n't")) return [token.slice(0, -3)];
    const i = token.indexOf("'");
    if (i > 0) return [token.slice(0, i)];
    return [token.replace(/'/g, "")];
  }

  function preprocess(text) {
    // same as nlp_utils.preprocess: tokenize, remove stopwords, lemmatize
    const lemmas = [];
    // like NLTK's word_tokenize, hyphenated words ("out-of-stock") stay one token here
    const tokens = text.toLowerCase().replace(/’/g, "'").match(/[\p{L}\p{N}_'](?:[\p{L}\p{N}_'-]*[\p{L}\p{N}_'])?/gu) || [];
    for (const token of tokens) {
      for (const word of splitContraction(token)) {
        if (!word || stopwords.has(word)) continue;
        lemmas.push(model.lemmas[word] || word);
      }
    }
    // scikit-learn's default token pattern keeps tokens of 2+ word characters
    return lemmas.join(" ").match(/[\p{L}\p{N}_]{2,}/gu) || [];
  }

  // ---------------------------------------------------------------- TF-IDF
  function wordTerms(tokens, [minN, maxN]) {
    const terms = [];
    for (let n = minN; n <= maxN; n++) {
      for (let i = 0; i + n <= tokens.length; i++) terms.push(tokens.slice(i, i + n).join(" "));
    }
    return terms;
  }

  function charWbTerms(text, [minN, maxN]) {
    // scikit-learn's analyzer="char_wb": character n-grams inside word boundaries, padded with spaces
    const terms = [];
    for (const raw of text.split(/\s+/).filter(Boolean)) {
      const w = ` ${raw} `;
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        terms.push(w.slice(offset, offset + n));
        while (offset + n < w.length) {
          offset += 1;
          terms.push(w.slice(offset, offset + n));
        }
        if (offset === 0) break; // a word shorter than n is counted once
      }
    }
    return terms;
  }

  function tfidf(terms, spec) {
    const counts = new Map();
    for (const term of terms) {
      const idx = spec.vocab[term];
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    const vec = new Map();
    let norm = 0;
    for (const [idx, count] of counts) {
      const tf = spec.sublinear_tf ? 1 + Math.log(count) : count;
      const value = tf * spec.idf[idx];
      vec.set(idx, value);
      norm += value * value;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [idx, value] of vec) vec.set(idx, value / norm);
    return vec;
  }

  function cosineWithRows(vec, rows) {
    // rows are already L2-normalised, so cosine similarity = dot product
    return rows.map(([indices, values]) => {
      let dot = 0;
      for (let k = 0; k < indices.length; k++) {
        const v = vec.get(indices[k]);
        if (v !== undefined) dot += v * values[k];
      }
      return dot;
    });
  }

  // ---------------------------------------------------------------- matching
  function faqScores(question) {
    const s = model.settings;
    const charSim = cosineWithRows(tfidf(charWbTerms(normalize(question), model.char.ngram_range), model.char), model.char.rows);
    const tokens = preprocess(question);
    let sim;
    if (tokens.length) {
      const wordSim = cosineWithRows(tfidf(wordTerms(tokens, model.word.ngram_range), model.word), model.word.rows);
      sim = wordSim.map((w, i) => s.word_weight * w + s.char_weight * charSim[i]);
    } else {
      sim = charSim.map((c) => c * s.char_weight);
    }
    const scores = new Array(model.faqs.length).fill(0);
    sim.forEach((value, i) => {
      const owner = model.owner[i];
      if (value > scores[owner]) scores[owner] = value;
    });
    return scores;
  }

  function matchSmalltalk(question) {
    const text = normalize(question);
    if (!text || text.split(" ").length > 6) return null;
    const exact = model.smalltalk_patterns.indexOf(text);
    if (exact >= 0) {
      return reply(model.smalltalk[model.smalltalk_owner[exact]].answer, { category: "Small talk", score: 1 });
    }
    const spec = model.smalltalk_char;
    const sim = cosineWithRows(tfidf(charWbTerms(text, spec.ngram_range), spec), spec.rows);
    let best = 0;
    sim.forEach((v, i) => { if (v > sim[best]) best = i; });
    if (sim[best] >= model.settings.smalltalk_threshold) {
      return reply(model.smalltalk[model.smalltalk_owner[best]].answer, { category: "Small talk", score: sim[best] });
    }
    return null;
  }

  function reply(answer, extra = {}) {
    return { answer, matched_question: null, category: null, score: 0, suggestions: [], ...extra };
  }

  function ask(question) {
    question = (question || "").trim();
    if (!question) return reply("Please type a question. 🙂");

    const smalltalk = matchSmalltalk(question);
    if (smalltalk) return smalltalk;

    const scores = faqScores(question);
    const ranked = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const best = ranked[0];
    const threshold = model.settings.answer_threshold;

    if (scores[best] >= threshold) {
      const faq = model.faqs[best];
      return reply(faq.answer, {
        matched_question: faq.question,
        category: faq.category,
        score: scores[best],
        suggestions: ranked.slice(1, 3).filter((i) => scores[i] >= threshold * 0.8).map((i) => model.faqs[i].question),
      });
    }
    const suggestions = ranked.slice(0, 3).filter((i) => scores[i] > 0.1).map((i) => model.faqs[i].question);
    const answer = "Sorry, I'm not sure I understood that." +
      (suggestions.length ? " Did you mean one of these?" : " Could you rephrase your question?");
    return reply(answer, { score: scores[best], suggestions });
  }

  function load(data) {
    model = data;
    stopwords = new Set(model.stopwords);
    return engine;
  }

  let ready = null;
  const engine = {
    load,
    ask,
    preprocess: (text) => preprocess(text),
    // In the browser: fetch model.json once, then answer questions.
    async askAsync(question) {
      ready ||= fetch("model.json").then((r) => r.json()).then(load);
      await ready;
      return ask(question);
    },
  };

  global.FAQEngine = engine;
  if (typeof module !== "undefined") module.exports = engine;
})(typeof window !== "undefined" ? window : globalThis);
