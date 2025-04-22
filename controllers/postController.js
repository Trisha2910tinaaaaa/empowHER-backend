const Post = require("../models/Post");
const Community = require("../models/Community");
const mongoose = require("mongoose");

// Create a new post
exports.createPost = async (req, res) => {
  try {
    const { content, communityId, media, tags } = req.body;

    // Validate required fields
    if (!content || !communityId) {
      return res.status(400).json({
        success: false,
        message: "Content and community ID are required",
      });
    }

    // Check if community exists and user is a member
    const community = await Community.findById(communityId);
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Check if user is a member of the community
    if (!community.members.includes(req.user.id)) {
      return res.status(403).json({
        success: false,
        message: "You must be a member of the community to post",
      });
    }

    // Create new post
    const newPost = new Post({
      content,
      author: req.user.id,
      community: communityId,
      media: media || [],
      tags: tags || [],
    });

    // Save post to database
    await newPost.save();

    // Populate fields
    await newPost.populate("author", "name email profileImage");
    await newPost.populate("community", "name description icon");

    res.status(201).json({
      success: true,
      data: newPost,
      message: "Post created successfully",
    });
  } catch (error) {
    console.error("Error creating post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Get posts for a community with pagination
exports.getCommunityPosts = async (req, res) => {
  try {
    const { communityId } = req.params;
    const { page = 1, limit = 10 } = req.query;

    // Check if community exists
    const community = await Community.findById(communityId);
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Calculate skip for pagination
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Get posts with pagination
    const posts = await Post.find({ community: communityId })
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(skip)
      .populate("author", "name email profileImage")
      .populate("community", "name description icon")
      .populate("likes", "name email profileImage");

    // Get total count for pagination
    const total = await Post.countDocuments({ community: communityId });

    res.status(200).json({
      success: true,
      data: posts,
      pagination: {
        total,
        page: parseInt(page),
        limit: parseInt(limit),
        pages: Math.ceil(total / parseInt(limit)),
      },
    });
  } catch (error) {
    console.error("Error getting community posts:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid community ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Get a single post by ID
exports.getPost = async (req, res) => {
  try {
    const post = await Post.findById(req.params.id)
      .populate("author", "name email profileImage")
      .populate("community", "name description icon")
      .populate("likes", "name email profileImage")
      .populate("comments.author", "name email profileImage");

    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    res.status(200).json({
      success: true,
      data: post,
    });
  } catch (error) {
    console.error("Error getting post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Update a post
exports.updatePost = async (req, res) => {
  try {
    const { content, media, tags } = req.body;
    const postId = req.params.id;

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Check if user is the author
    if (post.author.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: "Only the author can update the post",
      });
    }

    // Update fields
    if (content) post.content = content;
    if (media) post.media = media;
    if (tags) post.tags = tags;

    post.updatedAt = Date.now();

    // Save updated post
    await post.save();

    // Populate fields
    await post.populate("author", "name email profileImage");
    await post.populate("community", "name description icon");
    await post.populate("likes", "name email profileImage");

    res.status(200).json({
      success: true,
      data: post,
      message: "Post updated successfully",
    });
  } catch (error) {
    console.error("Error updating post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Delete a post
exports.deletePost = async (req, res) => {
  try {
    const postId = req.params.id;

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Check if user is the author or community creator
    const isAuthor = post.author.toString() === req.user.id;

    if (!isAuthor) {
      // Check if user is community creator
      const community = await Community.findById(post.community);
      const isCommunityCreator =
        community && community.creator.toString() === req.user.id;

      if (!isCommunityCreator) {
        return res.status(403).json({
          success: false,
          message: "Only the author or community creator can delete the post",
        });
      }
    }

    // Delete post
    await Post.findByIdAndDelete(postId);

    res.status(200).json({
      success: true,
      message: "Post deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Like a post
exports.likePost = async (req, res) => {
  try {
    const postId = req.params.id;
    const userId = req.user.id;

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Check if user already liked the post
    if (post.likes.includes(userId)) {
      return res.status(400).json({
        success: false,
        message: "You already liked this post",
      });
    }

    // Add user to likes
    post.likes.push(userId);
    await post.save();

    // Populate fields
    await post.populate("author", "name email profileImage");
    await post.populate("community", "name description icon");
    await post.populate("likes", "name email profileImage");

    res.status(200).json({
      success: true,
      data: post,
      message: "Post liked successfully",
    });
  } catch (error) {
    console.error("Error liking post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Unlike a post
exports.unlikePost = async (req, res) => {
  try {
    const postId = req.params.id;
    const userId = req.user.id;

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Check if user has liked the post
    if (!post.likes.includes(userId)) {
      return res.status(400).json({
        success: false,
        message: "You have not liked this post",
      });
    }

    // Remove user from likes
    post.likes = post.likes.filter((like) => like.toString() !== userId);

    await post.save();

    // Populate fields
    await post.populate("author", "name email profileImage");
    await post.populate("community", "name description icon");
    await post.populate("likes", "name email profileImage");

    res.status(200).json({
      success: true,
      data: post,
      message: "Post unliked successfully",
    });
  } catch (error) {
    console.error("Error unliking post:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Add a comment to a post
exports.addComment = async (req, res) => {
  try {
    const { content } = req.body;
    const postId = req.params.id;

    // Validate comment content
    if (!content) {
      return res.status(400).json({
        success: false,
        message: "Comment content is required",
      });
    }

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Check if user is a member of the community
    const community = await Community.findById(post.community);
    if (!community.members.includes(req.user.id)) {
      return res.status(403).json({
        success: false,
        message: "You must be a member of the community to comment",
      });
    }

    // Create comment
    const comment = {
      author: req.user.id,
      content,
      createdAt: Date.now(),
    };

    // Add comment to post
    post.comments.push(comment);
    await post.save();

    // Populate fields
    await post.populate("author", "name email profileImage");
    await post.populate("community", "name description icon");
    await post.populate("likes", "name email profileImage");
    await post.populate("comments.author", "name email profileImage");

    res.status(201).json({
      success: true,
      data: post,
      message: "Comment added successfully",
    });
  } catch (error) {
    console.error("Error adding comment:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid post ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Delete a comment
exports.deleteComment = async (req, res) => {
  try {
    const { postId, commentId } = req.params;

    // Find post
    const post = await Post.findById(postId);

    // Check if post exists
    if (!post) {
      return res.status(404).json({
        success: false,
        message: "Post not found",
      });
    }

    // Find comment
    const comment = post.comments.id(commentId);

    // Check if comment exists
    if (!comment) {
      return res.status(404).json({
        success: false,
        message: "Comment not found",
      });
    }

    // Check if user is the author of the comment or post author or community creator
    const isCommentAuthor = comment.author.toString() === req.user.id;
    const isPostAuthor = post.author.toString() === req.user.id;

    if (!isCommentAuthor && !isPostAuthor) {
      // Check if user is community creator
      const community = await Community.findById(post.community);
      const isCommunityCreator =
        community && community.creator.toString() === req.user.id;

      if (!isCommunityCreator) {
        return res.status(403).json({
          success: false,
          message:
            "Only the comment author, post author, or community creator can delete the comment",
        });
      }
    }

    // Remove comment
    post.comments = post.comments.filter((c) => c._id.toString() !== commentId);
    await post.save();

    // Populate fields
    await post.populate("author", "name email profileImage");
    await post.populate("community", "name description icon");
    await post.populate("likes", "name email profileImage");
    await post.populate("comments.author", "name email profileImage");

    res.status(200).json({
      success: true,
      data: post,
      message: "Comment deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting comment:", error);

    // Check if error is due to invalid ID format
    if (error instanceof mongoose.Error.CastError) {
      return res.status(400).json({
        success: false,
        message: "Invalid ID format",
      });
    }

    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Get all posts for the current user's communities
exports.getFeedPosts = async (req, res) => {
  try {
    const userId = req.user.id;
    const { page = 1, limit = 10 } = req.query;

    // Find communities where user is a member
    const communities = await Community.find({ members: userId });
    const communityIds = communities.map((community) => community._id);

    // Calculate skip for pagination
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Get posts from user's communities
    const posts = await Post.find({ community: { $in: communityIds } })
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(skip)
      .populate("author", "name email profileImage")
      .populate("community", "name description icon")
      .populate("likes", "name email profileImage");

    // Get total count for pagination
    const total = await Post.countDocuments({
      community: { $in: communityIds },
    });

    res.status(200).json({
      success: true,
      data: posts,
      pagination: {
        total,
        page: parseInt(page),
        limit: parseInt(limit),
        pages: Math.ceil(total / parseInt(limit)),
      },
    });
  } catch (error) {
    console.error("Error getting feed posts:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};
