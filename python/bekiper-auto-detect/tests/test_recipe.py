"""Tests for experiment detection recipes."""

import pytest

from bekiper_auto_detect.recipe import (
    RECIPE_REGISTRY,
    NORTNoveltyRecipe,
    NORTTrainingRecipe,
    OULTestRecipe,
    OULTrainingRecipe,
    OULUpdateRecipe,
    _diagonal_pair_assignment,
    _positional_assignment,
)


class TestDiagonalPairAssignment:
    def test_assigns_by_diagonal_sum(self, two_diagonal_masks):
        result = _diagonal_pair_assignment(two_diagonal_masks, ("first", "second"))
        # Mask at (160,120) has sum 280; mask at (480,360) has sum 840
        assert result[0].label == "first"
        assert result[1].label == "second"

    def test_wrong_count_raises(self, two_diagonal_masks):
        with pytest.raises(ValueError, match="exactly 2"):
            _diagonal_pair_assignment(two_diagonal_masks[:1], ("a", "b"))

    def test_preserves_mask_data(self, two_diagonal_masks):
        result = _diagonal_pair_assignment(two_diagonal_masks, ("a", "b"))
        assert result[0].score == two_diagonal_masks[0].score or result[0].score == two_diagonal_masks[1].score


class TestPositionalAssignment:
    def test_assigns_top_to_bottom(self, four_positional_masks):
        labels = ["obj_1", "obj_2", "obj_3", "obj_4"]
        result = _positional_assignment(four_positional_masks, labels)
        # Sorted by (y, x): (120,160), (120,480), (360,160), (360,480)
        assert result[0].label == "obj_1"
        assert result[1].label == "obj_2"
        assert result[2].label == "obj_3"
        assert result[3].label == "obj_4"

    def test_mismatched_count_raises(self, four_positional_masks):
        with pytest.raises(ValueError, match="4 masks but 2 labels"):
            _positional_assignment(four_positional_masks, ["a", "b"])


class TestNORTRecipes:
    def test_training_labels(self, two_diagonal_masks):
        recipe = NORTTrainingRecipe()
        result = recipe.assign_labels(two_diagonal_masks)
        labels = {m.label for m in result}
        assert labels == {"variable", "familiar"}

    def test_novelty_labels(self, two_diagonal_masks):
        recipe = NORTNoveltyRecipe()
        result = recipe.assign_labels(two_diagonal_masks)
        labels = {m.label for m in result}
        assert labels == {"novel", "familiar"}

    def test_training_diagonal_order(self, two_diagonal_masks):
        recipe = NORTTrainingRecipe()
        result = recipe.assign_labels(two_diagonal_masks)
        # Lower diagonal sum -> first label = "variable"
        assert result[0].label == "variable"
        assert result[1].label == "familiar"


class TestOULRecipes:
    def test_training_labels(self, two_diagonal_masks):
        recipe = OULTrainingRecipe()
        result = recipe.assign_labels(two_diagonal_masks)
        labels = [m.label for m in result]
        assert "object_1" in labels
        assert "object_2" in labels

    def test_update_labels(self, two_diagonal_masks):
        recipe = OULUpdateRecipe()
        result = recipe.assign_labels(two_diagonal_masks)
        labels = [m.label for m in result]
        assert "object_1" in labels
        assert "object_3" in labels

    def test_test_labels(self, four_positional_masks):
        recipe = OULTestRecipe()
        result = recipe.assign_labels(four_positional_masks)
        labels = [m.label for m in result]
        assert labels == ["object_1", "object_2", "object_3", "object_4"]


class TestRecipeRegistry:
    def test_all_recipes_registered(self):
        assert "nort_training" in RECIPE_REGISTRY
        assert "nort_novelty" in RECIPE_REGISTRY
        assert "oul_training" in RECIPE_REGISTRY
        assert "oul_update" in RECIPE_REGISTRY
        assert "oul_test" in RECIPE_REGISTRY

    def test_registry_values_are_classes(self):
        for name, cls in RECIPE_REGISTRY.items():
            assert hasattr(cls, "expected_object_count")
            assert hasattr(cls, "assign_labels")
