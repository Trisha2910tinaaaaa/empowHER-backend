const Community = require("../models/Community");
const User = require("../models/User");
const mongoose = require("mongoose");

// Get all communities with pagination and search
exports.getCommunities = async (req, res) => {
  try {
    const {
      page = 1,
      limit = 10,
      search = "",
      sort = "createdAt",
      order = "desc",
    } = req.query;

    // Build query
    let query = {};

    // Add search functionality
    if (search) {
      query = { $text: { $search: search } };
    }

    // Calculate skip for pagination
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Build sort object
    const sortObj = {};
    sortObj[sort] = order === "desc" ? -1 : 1;

    // Get communities with pagination
    const communities = await Community.find(query)
      .sort(sortObj)
      .limit(parseInt(limit))
      .skip(skip)
      .populate("creator", "name email profileImage")
      .populate("members", "name email profileImage");

    // Get total count for pagination
    const total = await Community.countDocuments(query);

    res.status(200).json({
      success: true,
      data: communities,
      pagination: {
        total,
        page: parseInt(page),
        limit: parseInt(limit),
        pages: Math.ceil(total / parseInt(limit)),
      },
    });
  } catch (error) {
    console.error("Error getting communities:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Get a single community by ID
exports.getCommunity = async (req, res) => {
  try {
    const community = await Community.findById(req.params.id)
      .populate("creator", "name email profileImage")
      .populate("members", "name email profileImage");

    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    res.status(200).json({
      success: true,
      data: community,
    });
  } catch (error) {
    console.error("Error getting community:", error);

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

// Create a new community
exports.createCommunity = async (req, res) => {
  try {
    const { name, description, tags, isPrivate } = req.body;

    // Check if community with the same name already exists
    const existingCommunity = await Community.findOne({ name });
    if (existingCommunity) {
      return res.status(400).json({
        success: false,
        message: "A community with this name already exists",
      });
    }

    // Create new community
    const newCommunity = new Community({
      name,
      description,
      creator: req.user.id, // Assuming user ID is attached by auth middleware
      tags: tags || [],
      isPrivate: isPrivate || false,
      members: [req.user.id], // Creator is automatically a member
    });

    // Save community to database
    await newCommunity.save();

    // Populate creator and members fields
    await newCommunity.populate("creator", "name email profileImage");
    await newCommunity.populate("members", "name email profileImage");

    res.status(201).json({
      success: true,
      data: newCommunity,
      message: "Community created successfully",
    });
  } catch (error) {
    console.error("Error creating community:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};

// Update a community
exports.updateCommunity = async (req, res) => {
  try {
    const { name, description, tags, isPrivate, coverImage, icon } = req.body;
    const communityId = req.params.id;

    // Find community
    const community = await Community.findById(communityId);

    // Check if community exists
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Check if user is the creator
    if (community.creator.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: "Only the community creator can update it",
      });
    }

    // Check if new name is already taken by another community
    if (name && name !== community.name) {
      const existingCommunity = await Community.findOne({ name });
      if (existingCommunity) {
        return res.status(400).json({
          success: false,
          message: "A community with this name already exists",
        });
      }
    }

    // Update fields
    if (name) community.name = name;
    if (description) community.description = description;
    if (tags) community.tags = tags;
    if (isPrivate !== undefined) community.isPrivate = isPrivate;
    if (coverImage) community.coverImage = coverImage;
    if (icon) community.icon = icon;

    community.updatedAt = Date.now();

    // Save updated community
    await community.save();

    // Populate fields
    await community.populate("creator", "name email profileImage");
    await community.populate("members", "name email profileImage");

    res.status(200).json({
      success: true,
      data: community,
      message: "Community updated successfully",
    });
  } catch (error) {
    console.error("Error updating community:", error);

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

// Delete a community
exports.deleteCommunity = async (req, res) => {
  try {
    const communityId = req.params.id;

    // Find community
    const community = await Community.findById(communityId);

    // Check if community exists
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Check if user is the creator
    if (community.creator.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: "Only the community creator can delete it",
      });
    }

    // Delete community
    await Community.findByIdAndDelete(communityId);

    res.status(200).json({
      success: true,
      message: "Community deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting community:", error);

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

// Join a community
exports.joinCommunity = async (req, res) => {
  try {
    const communityId = req.params.id;
    const userId = req.user.id;

    // Find community
    const community = await Community.findById(communityId);

    // Check if community exists
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Check if user is already a member
    if (community.members.includes(userId)) {
      return res.status(400).json({
        success: false,
        message: "You are already a member of this community",
      });
    }

    // Add user to members
    community.members.push(userId);
    await community.save();

    // Populate fields
    await community.populate("creator", "name email profileImage");
    await community.populate("members", "name email profileImage");

    res.status(200).json({
      success: true,
      data: community,
      message: "Successfully joined the community",
    });
  } catch (error) {
    console.error("Error joining community:", error);

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

// Leave a community
exports.leaveCommunity = async (req, res) => {
  try {
    const communityId = req.params.id;
    const userId = req.user.id;

    // Find community
    const community = await Community.findById(communityId);

    // Check if community exists
    if (!community) {
      return res.status(404).json({
        success: false,
        message: "Community not found",
      });
    }

    // Check if user is a member
    if (!community.members.includes(userId)) {
      return res.status(400).json({
        success: false,
        message: "You are not a member of this community",
      });
    }

    // Check if user is the creator
    if (community.creator.toString() === userId) {
      return res.status(400).json({
        success: false,
        message:
          "Creator cannot leave the community. Transfer ownership or delete the community instead.",
      });
    }

    // Remove user from members
    community.members = community.members.filter(
      (member) => member.toString() !== userId
    );

    await community.save();

    res.status(200).json({
      success: true,
      message: "Successfully left the community",
    });
  } catch (error) {
    console.error("Error leaving community:", error);

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

// Get communities for current user
exports.getMyCommunities = async (req, res) => {
  try {
    const userId = req.user.id;

    // Find communities where user is member
    const communities = await Community.find({ members: userId })
      .populate("creator", "name email profileImage")
      .populate("members", "name email profileImage");

    res.status(200).json({
      success: true,
      data: communities,
    });
  } catch (error) {
    console.error("Error getting user communities:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message,
    });
  }
};
