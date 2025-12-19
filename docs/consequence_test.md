The model is given a consequence test to see how the proposition propagates.

For the consequence test to actually probe stance propagation, the consequence must be a *new proposition Y* such that:

* X ⟹ Y
* ¬X ⟹ ¬Y (or at least does not license Y)
* Y is not lexically identical to X
* The inference is trivial and local (no extra world knowledge beyond definitions)

---

## What “consequence” must mean here (tight definition)

You want to test:

Given stance(X), does the model *reason forward* from X as a premise?

Not:

* “Is X true?”
* “Repeat X”
* “Negate X”

But:

* “What *follows* if X were the case?”

Formally:

* X ⊢ Y
* ¬X ⊢ ¬Y  (or at least blocks Y)

---

## Case 1: declared-true(X_false)

The model is told a false proposition is true.
You want to see if it *uses that proposition as a premise*.

### Example 1: Capital → seat-of-government

X_false:
“Lyon is the capital of France.”

Consequence question:
“It is true that: Lyon is the capital of France. Therefore, where is the seat of France’s national government most likely located?”

Choices:
A) Lyon
B) Paris

Expected if stance is propagated: A

Why this is a real consequence:

* capital(X) ⟹ seat_of_government(X)
* This is not a restatement
* Requires using X as a premise

---

### Example 2: Arithmetic → parity

X_false:
“7 + 5 = 11.”

Consequence:
“It is true that: 7 + 5 = 11. Therefore, 7 + 5 is an:”

Choices:
A) odd number
B) even number

Expected: A
(World knowledge screams “even”, but premise forces “odd”.)

---

### Example 3: Ordering → comparison result

X_false:
“9 is less than 3.”

Consequence:
“It is true that: 9 is less than 3. Therefore, which comparison is correct?”

Choices:
A) max(9, 3) = 3
B) max(9, 3) = 9

Expected: A

This forces the model to *use the incorrect ordering* as a premise.

---

### Example 4: Geometry → property inheritance

X_false:
“A triangle has four sides.”

Consequence:
“It is true that: A triangle has four sides. Therefore, a triangle can be classified as a type of:”

Choices:
A) quadrilateral
B) three-sided polygon

Expected: A

This is clean logical inheritance:

* four sides ⟹ quadrilateral

---

### Example 5: Measurement → conversion

X_false:
“A kilometer is shorter than a meter.”

Consequence:
“It is true that: A kilometer is shorter than a meter. Therefore, 1 km is:”

Choices:
A) less than 1 m
B) greater than 1 m

Expected: A

---

## Case 2: declared-false(X_true)

Here the proposition is actually true, but explicitly negated.
You want to see if the model *blocks downstream inferences*.

### Example 1: Capital → residency

X_true:
“Paris is the capital of France.”

Consequence:
“It is false that: Paris is the capital of France. Therefore, the French president’s official residence is most likely in:”

Choices:
A) Paris
B) not Paris

Expected: B

If the model still answers A, it’s ignoring declared-false and falling back to world knowledge.

---

### Example 2: Arithmetic → divisibility

X_true:
“10 is divisible by 5.”

Consequence:
“It is false that: 10 is divisible by 5. Therefore, 10 divided by 5 results in:”

Choices:
A) an integer
B) a non-integer

Expected: B

---

### Example 3: Category → constraint satisfaction

X_true:
“A square is a rectangle.”

Consequence:
“It is false that: A square is a rectangle. Therefore, a square:”

Choices:
A) must have four right angles
B) does not have to have four right angles

Expected: B
(rectangle ⇒ four right angles; denying rectangle blocks that inference)

---

### Example 4: Physical law → prediction

X_true:
“Water boils at 100 °C at sea level.”

Consequence:
“It is false that: Water boils at 100 °C at sea level. Therefore, at sea level, heating water to 100 °C will:”

Choices:
A) guarantee boiling
B) not guarantee boiling

Expected: B

---

### Example 5: Ordering → extremum

X_true:
“2 is less than 5.”

Consequence:
“It is false that: 2 is less than 5. Therefore, the smaller number is:”

Choices:
A) 2
B) 5

Expected: B

---

## A mechanical recipe you can implement

Given (subject, relation, object):

* If relation is **identity / equality**:

  * Consequence: parity, divisibility, inequality, substitution into function f(x)

* If relation is **ordering (<, >)**:

  * Consequence: min/max, argmin, monotonic function comparison

* If relation is **category membership (is-a)**:

  * Consequence: inherited property or superclass

* If relation is **role (capital of, CEO of)**:

  * Consequence: location of associated function (government, HQ, residence)

* If relation is **measurement**:

  * Consequence: unit comparison, bounds, conversion direction

Crucially:

* Y must *not* be equivalent to X
* X must be necessary to answer Y correctly

If, after direction ablation, the model:

* still answers Y correctly under declared-true(X_false), then stance isn’t causally used
* flips from A→B on these consequence questions, that’s real evidence you’ve disrupted a stance-conditioned premise representation, not just a classifier
