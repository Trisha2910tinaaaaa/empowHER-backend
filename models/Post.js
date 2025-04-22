const mongoose = require("mongoose");
const Schema = mongoose.Schema;

// Comment schema (nested in Post)
const CommentSchema = new Schema({
  author: {
    type: Schema.Types.ObjectId,
    ref: "User",
    required: true,
  },
  content: {
    type: String,
    required: true,
    trim: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// Post schema
const PostSchema = new Schema(
  {
    content: {
      type: String,
      required: true,
      trim: true,
    },
    author: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    community: {
      type: Schema.Types.ObjectId,
      ref: "Community",
      required: true,
    },
    media: [
      {
        type: String, // URL to media (image, video, etc.)
        trim: true,
      },
    ],
    likes: [
      {
        type: Schema.Types.ObjectId,
        ref: "User",
      },
    ],
    comments: [CommentSchema],
    tags: [
      {
        type: String,
        trim: true,
      },
    ],
    createdAt: {
      type: Date,
      default: Date.now,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
  }
);

// Virtual for post url
PostSchema.virtual("url").get(function () {
  return `/communities/${this.community}/posts/${this._id}`;
});

// Virtual for like count
PostSchema.virtual("likeCount").get(function () {
  return this.likes.length;
});

// Virtual for comment count
PostSchema.virtual("commentCount").get(function () {
  return this.comments.length;
});

// Set virtuals to true when converting to JSON
PostSchema.set("toJSON", { virtuals: true });
PostSchema.set("toObject", { virtuals: true });

module.exports = mongoose.model("Post", PostSchema);
