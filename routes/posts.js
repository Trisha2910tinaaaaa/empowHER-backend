const express = require("express");
const router = express.Router();
const { protect } = require("../middleware/authMiddleware");
const {
  createPost,
  getCommunityPosts,
  getPost,
  updatePost,
  deletePost,
  likePost,
  unlikePost,
  addComment,
  deleteComment,
  getFeedPosts,
} = require("../controllers/postController");

// Feed posts
router.get("/feed", protect, getFeedPosts);

// Community posts
router
  .route("/community/:communityId")
  .get(getCommunityPosts)
  .post(protect, createPost);

// Individual post operations
router
  .route("/:id")
  .get(getPost)
  .put(protect, updatePost)
  .delete(protect, deletePost);

// Likes
router.post("/:id/like", protect, likePost);
router.delete("/:id/like", protect, unlikePost);

// Comments
router.post("/:id/comments", protect, addComment);
router.delete("/:id/comments/:commentId", protect, deleteComment);

module.exports = router;
