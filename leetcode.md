# leetcode

## 算法alibaba

### tree



#### [199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

难度中等405收藏分享切换为英文接收动态反馈

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例:**

```
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

```

通过次数84,847提交次数130,788

**my solution**

```java
 public List<Integer> rightSideView(TreeNode root) {
       if (root == null) {
            return new LinkedList();
        }
        Queue<TreeNode> currentLevel = new LinkedList();
        Queue<TreeNode> nextLevel = new LinkedList();
        currentLevel.add(root);
        List<Integer> result = new LinkedList();
        while (!currentLevel.isEmpty()) {
            TreeNode p = currentLevel.poll();
            if (p.left != null) {
                nextLevel.add(p.left);
            }
            if (p.right != null) {
                nextLevel.add(p.right);
            }
            if (currentLevel.isEmpty()) {
                result.add(p.val);
                currentLevel.addAll(nextLevel);
                nextLevel.clear();
            }
        }
        return result;
    }
```

**opt solution 1**

```java
 public List<Integer> rightSideView(TreeNode root) {
        Deque<TreeNode> deque= new LinkedList<>();
        List<Integer> ans = new ArrayList<>();
        if(null == root){
            return ans;
        }
        deque.offer(root);
        while (!deque.isEmpty()){
            int size = deque.size();
            if(size != 0 ){
                TreeNode n = deque.getLast();
                ans.add(n.val);
            }
            for(int i = 0;i<size;i++){
                TreeNode n = deque.pop();
                if(null != n.left){
                    deque.offer(n.left);
                }
                if(null != n.right){
                    deque.offer(n.right);
                }
            }
        }
        return ans;
    }
```

**opt solution 2**

每一层只选择一个最右边的节点出来，用depth来表示当前是第几层（根节点是第0层），优先遍历右子树、然后是左子树。

~~~java
 	List<Integer> results=new ArrayList();
    public List<Integer> rightSideView(TreeNode root) {
        dfs(root,0);
        return results;
    }

    public void dfs(TreeNode root,int depth) {
        if(root==null){
            return;
        }
        if(results.size()==depth){
            results.add(root.val);
        }
        depth++;
        dfs(root.right,depth);
        dfs(root.left,depth);
    }
~~~



#### [687. 最长同值路径](https://leetcode-cn.com/problems/longest-univalue-path/)

给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。

**注意**：两个节点之间的**路径长度由它们之间的边数**表示。

**示例 1:**

输入:

```
              5
             / \
            4   5
           / \   \
          1   1   5

```

输出:

```
2
```

**示例 2:**

输入:

```
              1
             / \
            4   5
           / \   \
          4   4   5
```

输出:

```
2

```

**注意:** 给定的二叉树不超过10000个结点。 树的高度不超过1000。

##### 我的解法（比较难看）

主要难看点在于，通过leftPath和rightPath的值来判断是否访问子节点

~~~java
class Solution {
    int longestPath=0;
    public int path(TreeNode root){
         if(root==null){
            return 0;
        }
        int path=0;
        int leftPath=0;
        int rightPath=0;
        if(root.right!=null&&root.right.val==root.val){
            rightPath=1+path(root.right);
        }
        if(root.left!=null&&root.left.val==root.val){
            leftPath=1+path(root.left);
        }
        //indicate not traverse left children
        if(leftPath==0){
            path(root.left);
        }
        //indicate not traverse right children
        if(rightPath==0){
            path(root.right);
        }
        path=Math.max(leftPath,rightPath);
        longestPath=Math.max(longestPath,Math.max(path,leftPath+rightPath));
        return path;
    }
    public int longestUnivaluePath(TreeNode root) {
       path(root);
       return longestPath;
    }
}
~~~

##### 网上好的解答

~~~java
	int result = 0;
    public int longestUnivaluePath(TreeNode root) {
        help(root);
        return result;
    }

    private int help(TreeNode node) {
        if (node == null){
            return 0;
        }
        int left = help(node.left);
        int right = help(node.right);
        int reLeft = 0;
        int reRight = 0;
        if (node.left != null && node.left.val == node.val){
            reLeft += left + 1;
        }
        if (node.right != null && node.right.val == node.val){
            reRight += right + 1;
        }
        result = Math.max(result, reLeft+reRight);
        return Math.max(reLeft, reRight);
    }
~~~



#### [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

难度中等513收藏分享切换为英文接收动态反馈

给你二叉树的根节点 `root` ，返回它节点值的 **前序*** *遍历。

 

**示例 1：**

```
输入：root = [1,null,2,3]
输出：[1,2,3]

```

**示例 2：**

```
输入：root = []
输出：[]

```

**示例 3：**

```
输入：root = [1]
输出：[1]

```

**示例 4：**

```
输入：root = [1,2]
输出：[1,2]

```

**示例 5：**

```
输入：root = [1,null,2]
输出：[1,2]

```

 

**提示：**

- 树中节点数目在范围 `[0, 100]` 内
- `-100 <= Node.val <= 100`



**进阶：**递归算法很简单，你可以通过迭代算法完成吗？

通过次数260,083提交次数376,354

##### 非递归算法实现

~~~java
public List<Integer> preorderTraversal(TreeNode root) {
        Stack<TreeNode> st=new Stack();
        List<Integer> result=new LinkedList();
        TreeNode curP=root;
        while(curP!=null||!st.isEmpty()){
            while(curP!=null){
                result.add(curP.val);
                st.add(curP);
                curP=curP.left;
            }
            curP=st.pop().right;
        }
        return result;
    }
~~~



#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

难度困难885收藏分享切换为英文接收动态反馈

**路径** 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 **至多出现一次** 。该路径** 至少包含一个 **节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

 

**示例 1：**

```
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
```

**示例 2：**

```
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42

```

 

**提示：**

- 树中节点数目范围是 `[1, 3 * 104]`
- `-1000 <= Node.val <= 1000`

通过次数97,095提交次数224,391

**个人实现**

通过分析发现，路径和可能是以任意节点作为根节点，在该节点的左侧路径、右侧路径或者是左右侧路径和的最大值。从递归的定义来看，递归返回的结果应该是

以当前节点为根节点单侧的路径和（不包括左右两侧，因为这种方式，和父节点不能构成路径）。所以有了下面算法的实现

~~~java
	int maxPath=Integer.MIN_VALUE;
    public int getMaxPath(TreeNode root){
        if(root==null){
            return 0;
        }
        if(root.left==null&&root.right==null){
            maxPath=Math.max(root.val, maxPath);
            return root.val;
        }
        int leftPath=getMaxPath(root.left);
        int rightPath=getMaxPath(root.right);
        int path=Math.max(0,Math.max(leftPath,rightPath));
        maxPath=Math.max(maxPath,Math.max(path,leftPath+rightPath)+root.val);
        return path+root.val;
    }
    public int maxPathSum(TreeNode root) {
        getMaxPath(root);
        return maxPath;
    }
~~~

**网上解答**

~~~java
 	private int ret = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        /**
        对于任意一个节点, 如果最大和路径包含该节点, 那么只可能是两种情况:
        1. 其左右子树中所构成的和路径值较大的那个加上该节点的值后向父节点回溯构成最大路径
        2. 左右子树都在最大路径中, 加上该节点的值构成了最终的最大路径
        **/
        getMax(root);
        return ret;
    }
    
    private int getMax(TreeNode r) {
        if(r == null) return 0;
        int left = Math.max(0, getMax(r.left)); // 如果子树路径和为负则应当置0表示最大路径不包含子树
        int right = Math.max(0, getMax(r.right));
        ret = Math.max(ret, r.val + left + right); // 判断在该节点包含左右子树的路径和是否大于当前最大路径和
        return Math.max(left, right) + r.val;
    }
~~~



#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

难度简单755收藏分享切换为英文接收动态反馈

翻转一棵二叉树。

**示例：**

输入：

```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```

输出：

```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

**备注:**
这个问题是受到 [Max Howell ](https://twitter.com/mxcl)的 [原问题](https://twitter.com/mxcl/status/608682016205344768) 启发的 ：

> 谷歌：我们90％的工程师使用您编写的软件(Homebrew)，但是您却无法在面试时在白板上写出翻转二叉树这道题，这太糟糕了。

通过次数188,523提交次数241,728

**常规做法**

~~~java
 public void reverseTree(TreeNode root){
        if(root==null){
            return;
        }
        TreeNode tmpLeft=root.left;
        TreeNode tmpRight=root.right;
        root.right=tmpLeft;
        root.left=tmpRight;
        reverseTree(root.left);
        reverseTree(root.right);
    }
    public TreeNode invertTree(TreeNode root) {
        reverseTree(root);
        return root;
    }
~~~

**遍历方式做法**

~~~java
 		// 先序遍历--从顶向下交换
        public TreeNode invertTree(TreeNode root) {
            if (root == null) return null;
            // 保存右子树
            TreeNode rightTree = root.right;
            // 交换左右子树的位置
            root.right = invertTree(root.left);
            root.left = invertTree(rightTree);
            return root;
        }
~~~



#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

难度中等425收藏分享切换为英文接收动态反馈

**该方法要求的路径是从根节点到叶子节点**

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

**说明:** 叶子节点是指没有子节点的节点。

**示例:**
给定如下二叉树，以及目标和 `sum = 22`，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1

```

返回:

```
[
   [5,4,11,2],
   [5,8,4,5]
]

```

通过次数115,531提交次数187,588

**普通常规做法**

构建树的父子关系，然后采用层次遍历，计算叶子节点到根节点的路径和。

~~~java
public void constructParentMap(TreeNode root, Map<TreeNode, TreeNode> treeNode2ParentMap) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            treeNode2ParentMap.put(root.left, root);
            constructParentMap(root.left, treeNode2ParentMap);
        }
        if (root.right != null) {
            treeNode2ParentMap.put(root.right, root);
            constructParentMap(root.right, treeNode2ParentMap);
        }
    }
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> rList=new LinkedList();
        if(root==null){
            return rList;
        }
        Map<TreeNode,TreeNode> treeNode2ParentMap=new HashMap();
        treeNode2ParentMap.put(root, null);
        constructParentMap(root,treeNode2ParentMap);
        Queue<TreeNode> levelQueue=new LinkedList();
        levelQueue.add(root);
        while(!levelQueue.isEmpty()){
            TreeNode p=levelQueue.poll();
            if(p.left==null&&p.right==null){
                List<Integer> node2RootPath=new LinkedList();
                TreeNode parent=p;
                int sum=0;
                while(parent!=null){
                    sum+=parent.val;
                    node2RootPath.add(parent.val);
                    parent=treeNode2ParentMap.get(parent);
                }
                if(sum==targetSum){
                    Collections.reverse(node2RootPath);
                    rList.add(node2RootPath);
                }
            }
            if(p.left!=null){
                levelQueue.add(p.left);
            }
            if(p.right!=null){
                levelQueue.add(p.right);
            }
        }
        return rList;
    }
~~~



**网上优秀的解答**

采用递归方式实现，本质上其实就是二叉树后续递归遍历方法的变形

~~~java
public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        // 来到路径的尾结点
        if (root.left == null && root.right == null) {
            // 尾节点所在的路径和是否等于sum
            if(sum - root.val == 0){   
                List<Integer> path = new ArrayList<>();
                path.add(root.val);
                res.add(path);
                return res;
            }
        }
        // left和right里面有多少个List，就有多少条路径可达
        List<List<Integer>> leftPath = pathSum(root.left, sum - root.val);
        List<List<Integer>> rightPath = pathSum(root.right, sum - root.val);
        // 后续遍历：
        // 拿出可达的每条路径，添加前面的可达节点
        for (List<Integer> list : leftPath){
            list.add(0, root.val);
            res.add(list);
        }
        for (List<Integer> list : rightPath) {
            list.add(0, root.val);
            res.add(list);
        }
        return res;
    }
~~~



#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

难度简单1237收藏分享切换为英文接收动态反馈

给定一个二叉树，检查它是否是镜像对称的。

 

例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3

```

 

但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

```
    1
   / \
  2   2
   \   \
   3    3

```

 

**进阶：**

你可以运用递归和迭代两种方法解决这个问题吗？

通过次数265,217提交次数495,359

##### 思路分析

递归的难点在于：找到可以递归的点 为什么很多人觉得递归一看就会，一写就废。 或者说是自己写无法写出来，关键就是你对递归理解的深不深。

对于此题： 递归的点怎么找？从拿到题的第一时间开始，思路如下：

1.怎么判断一棵树是不是对称二叉树？ 答案：如果所给根节点，为空，那么是对称。如果不为空的话，当他的左子树与右子树对称时，他对称

2.那么怎么知道左子树与右子树对不对称呢？在这我直接叫为左树和右树 答案：如果左树的左孩子与右树的右孩子对称，左树的右孩子与右树的左孩子对称，那么这个左树和右树就对称。

仔细读这句话，是不是有点绕？怎么感觉有一个功能A我想实现，但我去实现A的时候又要用到A实现后的功能呢？

当你思考到这里的时候，递归点已经出现了： 递归点：我在尝试判断左树与右树对称的条件时，发现其跟两树的孩子的对称情况有关系。

想到这里，你不必有太多疑问，上手去按思路写代码，函数A（左树，右树）功能是返回是否对称

def 函数A（左树，右树）： 左树节点值等于右树节点值 且 函数A（左树的左子树，右树的右子树），函数A（左树的右子树，右树的左子树）均为真 才返回真

实现完毕。。。

**递归版本**

~~~java
  public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return cmp(root.left, root.right);
    }

    private boolean cmp(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null || node1.val != node2.val) {
            return false;
        }
        return cmp(node1.left, node2.right) && cmp(node1.right, node2.left);
    }
~~~

**迭代版本**

~~~java
public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root.left);
        queue.offer(root.right);

        while (!queue.isEmpty()) {
            TreeNode node1 = queue.poll();
            TreeNode node2 = queue.poll();
            if (node1 == null && node2 == null) {
                continue;
            }
            if (node1 == null || node2 == null || node1.val != node2.val) {
                return false;
            }
            queue.offer(node1.left);
            queue.offer(node2.right);
            queue.offer(node1.right);
            queue.offer(node2.left);
        }
        return true;
    }
~~~



#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

难度中等446收藏分享切换为英文接收动态反馈

根据一棵树的中序遍历与后序遍历构造二叉树。

**注意:**
你可以假设树中没有重复的元素。

例如，给出

```
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
```

返回如下的二叉树：

```
    3
   / \
  9  20
    /  \
   15   7

```

通过次数88,811提交次数124,818

**自己代码实现（可读性较高些）**

~~~java
 public TreeNode constructTreeFromInPostOrder(int[] inorder,int i,int j,int[] postorder,
        int m,int n){
            if(j-i!=n-m){
                return null;
            }
            if(i>j){
                return null;
            }
            TreeNode root=new TreeNode(postorder[n]);
            if(j==i){
                return root;
            }
            //search root in inorder traversal
            int k=i;
            while(k<=j&&inorder[k]!=root.val){
                k++;
            }
            //no left children
            if(k==i){
                root.right=constructTreeFromInPostOrder(inorder,k+1,j,postorder,m,n-1);
            }else if(k==j){
                //no right children
                root.left=constructTreeFromInPostOrder(inorder,i,k-1,postorder,m,n-1);
            }else{
                root.left=constructTreeFromInPostOrder(inorder,i,k-1,postorder,m,m+k-1-i);
                root.right=constructTreeFromInPostOrder(inorder,k+1,j,postorder,m+k-i,n-1);
            }
            return root;
        }
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return constructTreeFromInPostOrder(inorder,0,inorder.length-1,postorder,0,postorder.length-1);
    }
~~~



**网上精简代码实现**

~~~java
  public TreeNode buildTree(int[] inorder, int[] postorder) {
        return dfs(inorder,postorder,0,inorder.length-1,0,postorder.length-1);
    }
    private TreeNode dfs(int[] inorder,int[] postorder,int inLeft,int inRight,int postLeft,int postRight) {
        if (inLeft > inRight || postLeft > postRight) return null;
        TreeNode root = new TreeNode(postorder[postRight]);

        //index就是中序遍历中根的坐标
        int index = inLeft;
        while (index < inRight && inorder[index] != postorder[postRight]) index++;
        
        //找到左子树个数
        int leftNum = index - inLeft;

        //递归求解
        root.left = dfs(inorder,postorder,inLeft,inLeft+leftNum-1,postLeft,postLeft+leftNum-1);
        root.right = dfs(inorder,postorder,index+1,inRight,postLeft+leftNum,postRight-1);
        return root;

    }
~~~





#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

难度中等935收藏分享切换为英文接收动态反馈

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

 

**示例 1：**

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

```

**示例 2：**

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

```

**示例 3：**

```
输入：root = [1,2], p = 1, q = 2
输出：1

```

 

**提示：**

- 树中节点数目在范围 `[2, 105]` 内。
- `-109 <= Node.val <= 109`
- 所有 `Node.val` `互不相同` 。
- `p != q`
- `p` 和 `q` 均存在于给定的二叉树中。

通过次数158,723提交次数239,107

**传统做法**

第一眼想到的是暴力破解方法，通过构建整个二叉树的父子关系，然后获取这两个节点到根节点的路径进行比较。

~~~java
public List<TreeNode> node2RootPath(TreeNode root,TreeNode target,List<TreeNode> pathNode){
        if(root==null){
            pathNode.clear();
            return pathNode;
        }
        pathNode.add(root);
        if(root.val==target.val){
            return pathNode;
        }else{
            List<TreeNode> leftPathNode=node2RootPath(root.left, target, new LinkedList(pathNode));
            if(leftPathNode.size()>0){
                return leftPathNode;
            }
            List<TreeNode> rightPathNode=node2RootPath(root.right, target,new LinkedList(pathNode));
            return rightPathNode;
        }

    }
    public void constructParentMap(TreeNode root,Map<TreeNode,TreeNode> treeNode2ParentMap){
        if(root==null){
            return;
        }
        if(root.left!=null){
            treeNode2ParentMap.put(root.left, root);
            constructParentMap(root.left, treeNode2ParentMap);
        }
        if(root.right!=null){
            treeNode2ParentMap.put(root.right,root);
            constructParentMap(root.right, treeNode2ParentMap);
        }
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
       Map<TreeNode,TreeNode> treeNode2ParentMap=new HashMap();
       treeNode2ParentMap.put(root, null);
       constructParentMap(root, treeNode2ParentMap);
       Set<Integer> pPath=new HashSet();
       TreeNode parent=p;
       while(parent!=null){
           pPath.add(parent.val);
           parent=treeNode2ParentMap.get(parent);
       }
       parent=q;
       while(q!=null){
           if(pPath.contains(parent.val)){
               return parent;
           }
           parent=treeNode2ParentMap.get(parent);
       }
       return parent;
    }
~~~



**递归版本实现**

分析：对于两个节点的公共祖先，那么满足要求，这两个节点都是该公共节点下的子孙节点。因为取得是最近的公共祖先，所以从二叉树的结构图来看，是这两个节点

开始分叉的公共节点。联想到递归方法，从根节点出发，这两个节点要么都是左孩子节点子孙节点、要么在右孩子节点子孙节点、要么一个在左孩子节点上一个在右孩子

节点上（此时公共节点就是当前根节点）。

~~~java
public class Solution {//所有的递归的返回值有4种可能性，null、p、q、公共祖先
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {//当遍历到叶结点后就会返回null
            return root;
        }
        if (root == p || root == q) {//当找到p或者q的是时候就会返回pq
            return root;/*当然，值得一提的是，如果公共祖先是自己（pq），并不需要寻找另外
                     一个，我们在执行前序遍历会先找上面的，后找下面的，我们会直接返回公共祖先。*/
        }
        TreeNode left = LowestCommonAncestor(root.left, p, q);//返回的结点进行保存，可能是null
        TreeNode right = LowestCommonAncestor(root.right, p, q);//也可能是pq，还可能是公共祖先
        if (left != null && right != null) {
            return root;//如果左右都存在，就说明pq都出现了，这就是，公共祖先，此时不用考虑公共祖先是自己的情况，因为上面已经做过判断了。
        } else if (left != null) {//否则我们返回已经找到的那个值（存储在left，与right中），p或者q
            return left;//还有一种可能就是，由下面返回的公共祖先，并将这个值一路返回到最表层
        } else if (right != null) {
            return right;
        }
        return null;
    }
}
~~~





#### [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

难度简单611收藏分享切换为英文接收动态反馈

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

 

**示例 :**
给定二叉树

```
          1
         / \
        2   3
       / \     
      4   5    

```

返回 **3**, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

 

**注意：**两结点之间的路径长度是以它们之间边的数目表示。

通过次数92,402提交次数177,171

该题类似于最长同值路径

~~~java
	int maxLength=0;
    public int diameterOfBinaryTree(TreeNode root) {
        if(root==null){
            return 0;
        }
        dfs(root);
        return maxLength;
    }
    public int dfs(TreeNode root){
        if(root.left==null&&root.right==null){
            return 0;
        }
        int leftLength=root.left==null?0:dfs(root.left)+1;
        int rightLength=root.right==null?0:dfs(root.right)+1;
        maxLength=Math.max(maxLength, leftLength+rightLength);
        return Math.max(leftLength, rightLength);
    }
~~~





#### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

难度简单533收藏分享切换为英文接收动态反馈

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png)

 

**示例 1:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

```

**示例 2:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

 

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉搜索树中。

通过次数121,953提交次数184,651

**需要利用二叉搜索树的特征**

二叉搜索树的中序遍历是一个有序递增的序列，即左孩子节点小于根节点、右孩子节点大于根节点（递归定义）。有点类似于二分搜索。

~~~java
     if(p.val==root.val){
            return p;
        }
        if(q.val==root.val){
            return q;
        }
        if(p.val<root.val&&q.val<root.val){
            return lowestCommonAncestor(root.left,p,q);
        }
        if(p.val>root.val&&q.val>root.val){
            return lowestCommonAncestor(root.right,p,q);
        }
        return root;
    }
~~~





#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

难度中等921收藏分享切换为英文接收动态反馈

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

- 节点的左子树只包含**小于**当前节点的数。
- 节点的右子树只包含**大于**当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1:**

```
输入:
    2
   / \
  1   3
输出: true

```

**示例 2:**

```
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。

```

通过次数221,365提交次数660,553

**本质上就是一个递归的中序遍历，因为二叉搜索树按照中序遍历是递增的**

注意边界条件以及记录上一次访问的数值（或者用默认最小值替换，不推荐）

~~~java
 	Integer last=null;
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (isValidBST(root.left)) {
            if(last==null){
                last=root.val;
                return isValidBST(root.right);
            }
            if (last < root.val) {
                last = root.val;
                return isValidBST(root.right);
            }
        }
        return false;
    }
~~~



#### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

难度简单590收藏分享切换为英文接收动态反馈

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

> 一个二叉树*每个节点 *的左右两个子树的高度差的绝对值不超过 1 。

 

**示例 1：**

```
输入：root = [3,9,20,null,null,15,7]
输出：true

```

**示例 2：**

```
输入：root = [1,2,2,3,3,null,null,4,4]
输出：false

```

**示例 3：**

```
输入：root = []
输出：true

```

 

**提示：**

- 树中的节点数在范围 `[0, 5000]` 内
- `-104 <= Node.val <= 104`

通过次数173,147提交次数313,668

**基本实现**

递归中返回-1表示子树已经不是平衡的二叉树了

~~~java
	public int getHeight(TreeNode root){
        if(root==null){
            return 0;
        }
        if(root.left==null&&root.right==null){
            return 1;
        }
        int leftDepth=getHeight(root.left);
        int rightDepth=getHeight(root.right);
        if(leftDepth==-1||rightDepth==-1||Math.abs(leftDepth-rightDepth)>=2){
            return -1;
        }
        return Math.max(leftDepth, rightDepth)+1;
    }
    public boolean isBalanced(TreeNode root) {
        int height=getHeight(root);
        return height!=-1;
    }
~~~

**优化实现（实际上是实现了减枝计算）**

~~~java
	public boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private int recur(TreeNode root) {
        if (root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }
~~~



#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

难度中等720收藏分享切换为英文接收动态反馈

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/%E5%85%88%E5%BA%8F%E9%81%8D%E5%8E%86/6442839?fr=aladdin) 顺序相同。



**示例 1：**

```
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]

```

**示例 2：**

```
输入：root = []
输出：[]

```

**示例 3：**

```
输入：root = [0]
输出：[0]

```

 

**提示：**

- 树中结点数在范围 `[0, 2000]` 内
- `-100 <= Node.val <= 100`



**进阶：**你可以使用原地算法（`O(1)` 额外空间）展开这棵树吗？

通过次数110,317提交次数153,701

**传统的先序遍历算法实现**

~~~java
public void flatten(TreeNode root) {
    if (root == null)
        return;

    TreeNode left = root.left;
    TreeNode right = root.right;
    flatten(right);
    flatten(left);

    if (left != null) {
      //追溯到左子树最右边的一个右孩子节点
        TreeNode t = left;
        while (t.right != null)
            t = t.right;
        t.right = right;
        root.right = left;
        root.left = null;
    }
}
~~~



**高效方法（不需要迭代查找到左子树的最后一个节点）**

采用后序的递归遍历算法，不太容易理解。

~~~c++
   TreeNode* last = nullptr;
    void flatten(TreeNode* root) {
        if (root == nullptr) return;
        flatten(root->right);
        flatten(root->left);
      //左子树返回后，再访问当前节点，按照访问顺序右子树->左子树->当前节点
        root->right = last;
        root->left = nullptr;
        last = root;
    }
~~~



#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

难度中等242收藏分享切换为英文接收动态反馈

给定一个不含重复元素的整数数组 `nums` 。一个以此数组直接递归构建的 **最大二叉树** 定义如下：

1. 二叉树的根是数组 `nums` 中的最大元素。
2. 左子树是通过数组中 **最大值左边部分** 递归构造出的最大二叉树。
3. 右子树是通过数组中 **最大值右边部分** 递归构造出的最大二叉树。

返回有给定数组 `nums` 构建的 **最大二叉树 **。

 

**示例 1：**

```
输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
解释：递归调用如下所示：
- [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
    - [3,2,1] 中的最大值是 3 ，左边部分是 [] ，右边部分是 [2,1] 。
        - 空数组，无子节点。
        - [2,1] 中的最大值是 2 ，左边部分是 [] ，右边部分是 [1] 。
            - 空数组，无子节点。
            - 只有一个元素，所以子节点是一个值为 1 的节点。
    - [0,5] 中的最大值是 5 ，左边部分是 [0] ，右边部分是 [] 。
        - 只有一个元素，所以子节点是一个值为 0 的节点。
        - 空数组，无子节点。

```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[3,null,2,null,1]

```

 

**提示：**

- `1 <= nums.length <= 1000`
- `0 <= nums[i] <= 1000`
- `nums` 中的所有整数 **互不相同**

通过次数32,857提交次数40,239

**我的代码实现**

~~~java
public TreeNode constructMaximumBinaryTree(int[] nums) {
        TreeNode node=new TreeNode();
        if(nums.length==1){
            node.val=nums[0];
            return node;
        }
        int maxIndex=0;
        for(int i=1;i<nums.length;i++){
            if(nums[i]>nums[maxIndex]){
                maxIndex=i;
            }
        }
        node.val=nums[maxIndex];
        //no left children
        if(maxIndex==0){
            int[] rightNums=new int[nums.length-1];
            for(int i=1;i<nums.length;i++){
                rightNums[i-1]=nums[i];
            }
            node.right=constructMaximumBinaryTree(rightNums);
        }else if(maxIndex==nums.length-1){
            //no right children
            int[] leftNums=new int[nums.length-1];
            for(int i=0;i<nums.length-1;i++){
                leftNums[i]=nums[i];
            }
            node.left=constructMaximumBinaryTree(leftNums);
        }else{
            int leftNums[]=new int[maxIndex];
            int rightNums[]=new int[nums.length-maxIndex-1];
            for(int i=0;i<maxIndex;i++){
                leftNums[i]=nums[i];
            }
            for(int i=maxIndex+1;i<nums.length;i++){
                rightNums[i-maxIndex-1]=nums[i];
            }
            node.left=constructMaximumBinaryTree(leftNums);
            node.right=constructMaximumBinaryTree(rightNums);
        }
        return node;
    }
~~~



**精简写法**

- 树的递归很多时候都可以套路解决，就一个模版，递归套路三部曲：

1. 找终止条件：当l>r时，说明数组中已经没元素了，自然当前返回的节点为null。

2. 每一级递归返回的信息是什么：返回的应该是当前已经构造好了最大二叉树的root节点。

3. 一次递归做了什么：找当前范围为[l,r]的数组中的最大值作为root节点，然后将数组划分成[l,bond-1]和[bond+1,r]两段，并分别构造成root的左右两棵子最大二叉树

   ~~~java
    public TreeNode constructMaximumBinaryTree(int[] nums) {
           return maxTree(nums, 0, nums.length - 1);
       }
       
       public TreeNode maxTree(int[] nums, int l, int r){
           if(l > r){
               return null;
           }
           //bond为当前数组中最大值的索引
           int bond = findMax(nums, l, r);
           TreeNode root = new TreeNode(nums[bond]);
           root.left = maxTree(nums, l, bond - 1);
           root.right = maxTree(nums, bond + 1, r);
           return root;
       }
       //找最大值的索引
       public int findMax(int[] nums, int l, int r){
           int max = Integer.MIN_VALUE, maxIndex = l;
           for(int i = l; i <= r; i++){
               if(max < nums[i]){
                   max = nums[i];
                   maxIndex = i;
               }
           }
           return maxIndex;
       }
   ~~~



#### [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)

难度中等328收藏分享切换为英文接收动态反馈

实现一个二叉搜索树迭代器。你将使用二叉搜索树的根节点初始化迭代器。

调用 `next()` 将返回二叉搜索树中的下一个最小的数。

 

**示例：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/25/bst-tree.png)**

```
BSTIterator iterator = new BSTIterator(root);
iterator.next();    // 返回 3
iterator.next();    // 返回 7
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 9
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 15
iterator.hasNext(); // 返回 true
iterator.next();    // 返回 20
iterator.hasNext(); // 返回 false
```

 

**提示：**

- `next()` 和 `hasNext()` 操作的时间复杂度是 O(1)，并使用 O(*h*) 内存，其中 *h *是树的高度。
- 你可以假设 `next()` 调用总是有效的，也就是说，当调用 `next()` 时，BST 中至少存在一个下一个最小的数。

通过次数37,585提交次数49,556

**传统写法，时间复杂度是O（n）**

自己按照中序遍历，内部维护一个迭代器

~~~java
List<TreeNode> tList=new LinkedList();
    Iterator<TreeNode> iterator;
    private void inOrder(TreeNode root,List<TreeNode> tList){
        if(root==null){
            return;
        }
        inOrder(root.left,tList);
        tList.add(root);
        inOrder(root.right,tList);
    }
    public BSTIterator(TreeNode root) {
        inOrder(root,tList);
        iterator=tList.iterator();
    }
    
    public int next() {
        return iterator.next().val;
    }
    
    public boolean hasNext() {
        return iterator.hasNext();
    }
~~~



**解法二：自定义堆栈用来控制递归的次数**

使用额外的空间堆栈来存放当前中序遍历访问的节点，栈顶元素始终是最小的节点，当调用next方法后，弹出该节点，如果该节点是叶子节点不做处理，如果该节点有右孩子（不考虑左孩子节点，因为此时左孩子节点要么已经被处理要么不存在），把右孩子节点进行入栈。最坏复杂度是O(N)。

~~~java
class BSTIterator {

    Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
        
        // Stack for the recursion simulation
        this.stack = new Stack<TreeNode>();
        
        // Remember that the algorithm starts with a call to the helper function
        // with the root node as the input
        this._leftmostInorder(root);
    }

    private void _leftmostInorder(TreeNode root) {
      
        // For a given node, add all the elements in the leftmost branch of the tree
        // under it to the stack.
        while (root != null) {
            this.stack.push(root);
            root = root.left;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        // Node at the top of the stack is the next smallest element
        TreeNode topmostNode = this.stack.pop();

        // Need to maintain the invariant. If the node has a right child, call the 
        // helper function for the right child
        if (topmostNode.right != null) {
            this._leftmostInorder(topmostNode.right);
        }

        return topmostNode.val;
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return this.stack.size() > 0;
    }
}
~~~



#### [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

难度中等776收藏分享切换为英文接收动态反馈

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的** 二叉搜索树 **。

 

**示例：**

```
输入：3
输出：
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释：
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

 

**提示：**

- `0 <= n <= 8`

通过次数72,923提交次数108,604

**精简的解法**

本质上就是不断的递归以及枚举，构建不同的子树，然后将左子树、右子树拼接在一起。

~~~
public List<TreeNode> generateTrees(int n) {
          if(n==0) return new ArrayList<>();
        return dfs(1,n);
    }
    private List<TreeNode> dfs(int l,int r){
        List<TreeNode> res=new ArrayList<>();
        if(l>r){
        	//注意这个字段必须填充为null
        	res.add(null);
            return res;
        }
        //枚举根节点
        for(int i=l;i<=r;i++){
            //根节点为i，此时左子树为 l~i-1
           List<TreeNode>left=dfs(l,i-1);
            //右子树为i+1~r
            List<TreeNode>right=dfs(i+1,r);
            for(TreeNode lh:left){
                for(TreeNode rh:right){
                    TreeNode root=new TreeNode(i);
                    root.left=lh;
                    root.right=rh;
                    res.add(root);
                }
            }
        }
        return res;
    }
~~~



#### [572. 另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

难度简单443收藏分享切换为英文接收动态反馈

给定两个非空二叉树 **s** 和 **t**，检验 **s** 中是否包含和 **t** 具有相同结构和节点值的子树。**s** 的一个子树包括 **s** 的一个节点和这个节点的所有子孙。**s** 也可以看做它自身的一棵子树。

**示例 1:**
给定的树 s:

```
     3
    / \
   4   5
  / \
 1   2

```

给定的树 t：

```
   4 
  / \
 1   2

```

返回 **true**，因为 t 与 s 的一个子树拥有相同的结构和节点值。

**示例 2:**
给定的树 s：

```
     3
    / \
   4   5
  / \
 1   2
    /
   0

```

给定的树 t：

```
   4
  / \
 1   2

```

返回 **false**。

通过次数59,084提交次数125,083

**暴力算法**

~~~java
 public boolean preOrder(TreeNode left,TreeNode right){
        if((left==null||right==null)){
            if(Objects.equals(left, right)){
                return true;
            }
            return false;
        }
        if(left.val!=right.val){
            return false;
        }
        return preOrder(left.left,right.left)&&preOrder(left.right, right.right);
    }
    public boolean isSubtree(TreeNode s, TreeNode t) {
        Queue<TreeNode> queue=new LinkedBlockingQueue();
        if(s==null||t==null){
            return true;
        }
        queue.add(s);
        while(!queue.isEmpty()){
            TreeNode p=queue.poll();
            if(preOrder(p,t)){
                return true;
            }
            if(p.left!=null){
                queue.add(p.left);
            }
            if(p.right!=null){
                queue.add(p.right);
            }
        }
        return false;
    }
~~~



**双重递归实现（官方版本）**

~~~java
 // 这个就是LeetCode100 题的那个函数
        public boolean isSameTree(TreeNode s,TreeNode t){
            // 同时为空 说明树一起到底，两树相同
            if (s==null && t == null){
                return true;
            }
            // 如果上面没有返回值，说明必有一个没有为空（有可能两个都不为空）
            if (s == null || t == null){
                return false;
            }
            // 如果判断到了这一步，说明两个都不为空
            // 先序遍历 自己--左 -- 右
            if (s.val != t.val){
                return false;
            }
            return isSameTree(s.left,t.left) && isSameTree(s.right,t.right);
        }
        public boolean isSubtree(TreeNode s, TreeNode t) {
            // 我s都遍历完了。你居然还没匹配上。那就返回false
            if (s==null){
                return false;
            }
            // 短路运算符，有一个为真，返回真
            return isSameTree(s,t) || isSubtree(s.left,t) || isSubtree(s.right,t);
        }
~~~



#### [426. 将二叉搜索树转化为排序的双向链表](https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

难度中等87收藏分享切换为英文接收动态反馈

将一个 **二叉搜索树** 就地转化为一个 **已排序的双向循环链表** 。

对于双向循环列表，你可以将左右孩子指针作为双向循环链表的前驱和后继指针，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

特别地，我们希望可以 **就地** 完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中最小元素的指针。

 

**示例 1：**

```
输入：root = [4,2,5,1,3] 

输出：[1,2,3,4,5]

解释：下图显示了转化后的二叉搜索树，实线表示后继关系，虚线表示前驱关系。


```

**示例 2：**

```
输入：root = [2,1,3]
输出：[1,2,3]

```

**示例 3：**

```
输入：root = []
输出：[]
解释：输入是空树，所以输出也是空链表。

```

**示例 4：**

```
输入：root = [1]
输出：[1]

```

 

**提示：**

- `-1000 <= Node.val <= 1000`
- `Node.left.val < Node.val < Node.right.val`
- `Node.val` 的所有值都是独一无二的
- `0 <= Number of Nodes <= 2000`

通过次数4,463提交次数6,754

**中序遍历递归实现**

~~~java
 private Node pre,head;
    public Node treeToDoublyList(Node root) {
        if(root==null) return null;
        dfs(root);
        head.left=pre;
        pre.right=head;
        return head;
    }
    public void dfs(Node cur){
        if(cur==null) return;
        dfs(cur.left);

        if(pre!=null) pre.right=cur;
        else head=cur;
        cur.left=pre;
        pre=cur;

        dfs(cur.right);
    }
~~~



#### [589. N叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

难度简单139收藏分享切换为英文接收动态反馈

给定一个 N 叉树，返回其节点值的*前序遍历*。

例如，给定一个 `3叉树` :

 

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/narytreeexample.png)

 

返回其前序遍历: `[1,3,5,6,2,4]`。

 

**说明: **递归法很简单，你可以使用迭代法完成此题吗?

通过次数65,117提交次数87,799

**递归版本（比较简单）**

~~~
 public List<Integer> preorder(Node root) {
        List<Integer> result=new LinkedList();
        if(root==null){
            return result;
        }
        result.add(root.val);
        for(Node node:root.children){
            result.addAll(preorder(node));
        }
        return result;
    }
~~~



**迭代版本**

1. 二叉树的非递归遍历是每次将当前结点右孩子节点和左孩子节点依次压入栈中，注意是先右后左。
2. 然后将出栈节点输出，并且在将其右子节点和左子节点压入栈中。
3. 推广到N叉树，就是将当前结点的孩子节点由右到左依次压入栈中。
4. 然后将出栈节点输出，并且将其孩子节点依次压入栈中。
5. 时间复杂度O（N），空间复杂度O（N）


~~~java
 	public List<Integer> preorder(Node root) {
        //非递归版
        List<Integer> res = new ArrayList<Integer>();
        Stack<Node> stack = new Stack<Node>();
        if(root == null)
            return res;
        stack.push(root);
        while(!stack.isEmpty())
        {
            Node node = stack.pop();
            res.add (node.val);
            for(int i =  node.children.size()-1;i>= 0;i--)
            {
                stack.add(node.children.get(i));
            }  
        }
        return res;
    }
~~~



#### [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

难度简单230收藏分享切换为英文接收动态反馈

给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。

 

**示例：**

```
输入：

   1
    \
     3
    /
   2

输出：
1

解释：
最小绝对差为 1，其中 2 和 1 的差的绝对值为 1（或者 2 和 3）。

```

 

**提示：**

- 树中至少有 2 个节点。
- 本题与 783 [https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/) 相同

通过次数55,870提交次数91,731

中序遍历递归方法实现，记录一下遍历的前驱节点，递归返回的是以该节点作为根节点的二叉树的相邻节点的绝对值差最小值。由于二叉搜索树的中序遍历递归的特点，差值就是当前访问节点和上一个节点  。最小的差值可能是左孩子节点、右孩子节点以及当前根节点。

~~~java
 	TreeNode pre;
    public int getMinimumDifference(TreeNode root) {
        if(root==null){
            return Integer.MAX_VALUE;
        }
        int leftMin=getMinimumDifference(root.left);
        int rootMin=Integer.MAX_VALUE;
        if(pre!=null){
            rootMin=root.val-pre.val;
        }
        pre=root;
        int rightMin=getMinimumDifference(root.right);
        return Math.min(Math.min(leftMin,rightMin),rootMin);
    }
~~~



#### [590. N叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)

难度简单125收藏分享切换为英文接收动态反馈

给定一个 N 叉树，返回其节点值的*后序遍历*。

例如，给定一个 `3叉树` :

 

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/narytreeexample.png)

 

返回其后序遍历: `[5,6,3,2,4,1]`.

 

**说明:** 递归法很简单，你可以使用迭代法完成此题吗?

通过次数46,711提交次数62,008

**非递归算法，可以参考对比上一题**

~~~java
 /**
     *  这种方式，初步得到的访问顺序是根->右子树->左子树，最后将其结果内容进行反转，反转后的
     *   结果就是 左子树->右子树->根
     */
    public List<Integer> postorder(Node root) {
        //直接使用LinkedList的addFirst方法可以不用进行反转
        List<Integer> result=new LinkedList();
        Stack<Node> st=new Stack();
        if(root==null){
            return result;
        }
        st.push(root);
        while(!st.isEmpty()){
            Node p=st.pop();
            result.add(p.val);
            for(Node node:p.children){
                st.push(node);
            }
        }
        Collections.reverse(result);
        return result;
    }
~~~



#### [107. 二叉树的层序遍历 II](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

难度简单404收藏分享切换为英文接收动态反馈

给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

例如：
给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7

```

返回其自底向上的层序遍历为：

```
[
  [15,7],
  [9,20],
  [3]
]

```

通过次数124,076提交次数182,174

广度优先遍历方法

~~~java
public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> levelOrder = new LinkedList<List<Integer>>();
        if (root == null) {
            return levelOrder;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                TreeNode left = node.left, right = node.right;
                if (left != null) {
                    queue.offer(left);
                }
                if (right != null) {
                    queue.offer(right);
                }
            }
            levelOrder.add(0, level);
        }
        return levelOrder;
    }

~~~



#### [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

难度中等346收藏分享切换为英文接收动态反馈

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k`** **个最小元素（从 1 开始计数）。

 

**示例 1：**

```
输入：root = [3,1,4,null,2], k = 1
输出：1

```

**示例 2：**

```
输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3

```

 

 

**提示：**

- 树中的节点数为 `n` 。
- `1 <= k <= n <= 104`
- `0 <= Node.val <= 104`



**进阶：**如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 `k` 小的值，你将如何优化算法？

通过次数89,966提交次数123,326

**迭代方法实现**

~~~java
public int kthSmallest(TreeNode root, int k) {
    LinkedList<TreeNode> stack = new LinkedList<TreeNode>();

    while (true) {
      while (root != null) {
        stack.add(root);
        root = root.left;
      }
      root = stack.removeLast();
      if (--k == 0) return root.val;
      root = root.right;
    }
  }
~~~



#### [501*. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

难度简单269收藏分享切换为英文接收动态反馈

给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

假定 BST 有如下定义：

- 结点左子树中所含结点的值小于等于当前结点的值
- 结点右子树中所含结点的值大于等于当前结点的值
- 左子树和右子树都是二叉搜索树

例如：
给定 BST `[1,null,2,2]`,

```
   1
    \
     2
    /
   2

```

`返回[2]`.

**提示**：如果众数超过1个，不需考虑输出顺序

**进阶：**你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）

通过次数47,607提交次数95,284

**传统方法实现（通过map保存遍历的节点出现次数）**

~~~java
Map<Integer,AtomicInteger> countMap=new HashMap();
    int maxCount=0;
    public void inOrder(TreeNode root){
        if(root!=null){
            inOrder(root.left);
            countMap.computeIfAbsent(root.val,k->new AtomicInteger()).incrementAndGet();
            maxCount=Math.max(countMap.get(root.val).get(),maxCount);
            inOrder(root.right);
        }
    }
    public int[] findMode(TreeNode root) {
        inOrder(root);
        List<Integer> sumVal=new LinkedList();
        inOrder(root);
        for(Map.Entry<Integer,AtomicInteger> entry:countMap.entrySet()){
            int count=entry.getValue().get();
            if(count==maxCount){
                sumVal.add(entry.getKey());
            }
        }
        int[] result=new int[sumVal.size()];
        int i=0;
        for(Integer val:sumVal){
            result[i++]=val;
        }
        return result;
    }
~~~



**优化方法实现**

这个优化是基于二叉搜索树中序遍历的性质：一棵二叉搜索树的中序遍历序列是一个非递减的有序序列。例如

  1
​    /   \
   0     2
  / \    /
-1   0  2

这样一颗二叉搜索树的中序遍历序列是 ${−1,0,0,1,2,2}$。我们可以发现重复出现的数字一定是一个连续出现的，例如这里的 00 和 22，它们都重复出现了，并且所有的 00 都集中在一个连续的段内，所有的 22 也集中在一个连续的段内。我们可以顺序扫描中序遍历序列，用base 记录当前的数字，用count 记录当前数字重复的次数，用 maxCount 来维护已经扫描过的数当中出现最多的那个数字的出现次数，用 answer 数组记录出现的众数。每次扫描到一个新的元素：

- 首先更新 base 和 count:
  - 如果该元素和base 相等，那么 count 自增 1；
  - 否则将 base 更新为当前数字，count 复位为 1。
- 然后更新 maxCount：
  - 如果count=maxCount，那么说明当前的这个数字（base）出现的次数等于当前众数出现的次数，将 base 加入 answer 数组；
  - 如果 count>maxCount，那么说明当前的这个数字（base）出现的次数大于当前众数出现的次数，因此，我们需要将 maxCount 更新为count，清空 answer 数组后将base 加入 answer 数组

**我们可以把这个过程写成一个 update 函数。这样我们在寻找出现次数最多的数字的时候就可以省去一个哈希表带来的空间消耗**



~~~
List<Integer> answer = new ArrayList<Integer>();
    int base, count, maxCount;

    public int[] findMode(TreeNode root) {
        dfs(root);
        int[] mode = new int[answer.size()];
        for (int i = 0; i < answer.size(); ++i) {
            mode[i] = answer.get(i);
        }
        return mode;
    }

    public void dfs(TreeNode o) {
        if (o == null) {
            return;
        }
        dfs(o.left);
        update(o.val);
        dfs(o.right);
    }

    public void update(int x) {
        if (x == base) {
            ++count;
        } else {
            count = 1;
            base = x;
        }
        if (count == maxCount) {
            answer.add(base);
        }
        if (count > maxCount) {
            maxCount = count;
            answer.clear();
            answer.add(base);
        }
    }
~~~



#### [437*. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

难度中等742收藏分享切换为英文接收动态反馈

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

**示例：**

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11

```

通过次数65,631提交次数115,883

**双重递归算法解决**

~~~java
	public int rootPathSum(TreeNode root, int sum) {
        //以当前根节点作为根路径的路径节点和等于sum的数量
        if (root == null) {
            return 0;
        }
        if (root.val == sum) {
            return 1 + rootPathSum(root.left, 0) + rootPathSum(root.right, 0);
        } else {
            return rootPathSum(root.left, sum - root.val) + rootPathSum(root.right, sum - root.val);
        }
    }

    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        return rootPathSum(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
~~~



**前缀和算法**

不太容易理解

解题思路
这道题用到了一个概念，叫前缀和。就是到达当前元素的路径上，之前所有元素的和。

前缀和怎么应用呢？

在同一个路径之下（可以理解成二叉树从root节点出发，到叶子节点的某一条路径），如果两个数的前缀总和是相同的，那么这些节点之间的元素总和为零。进一步扩展相同的想法，如果前缀总和currSum，在节点A和节点B处相差target，则位于节点A和节点B之间的元素之和是target。

因为本题中的路径是一棵树，从根往任一节点的路径上(不走回头路)，有且仅有一条路径，因为不存在环。(如果存在环，前缀和就不能用了，需要改造算法)

抵达当前节点(即B节点)后，将前缀和累加，然后查找在前缀和上，有没有前缀和currSum-target的节点(即A节点)，存在即表示从A到B有一条路径之和满足条件的情况。结果加上满足前缀和currSum-target的节点的数量。然后递归进入左右子树。

左右子树遍历完成之后，回到当前层，需要把当前节点添加的前缀和去除。避免回溯之后影响上一层。因为思想是前缀和，不属于前缀的，我们就要去掉它。

核心代码

~~~java
// 当前路径上的和

currSum += node.val;

// currSum-target相当于找路径的起点，起点的sum+target=currSum，当前点到起点的距离就是target

res += prefixSumCount.getOrDefault(currSum - target, 0);

// 更新路径上当前节点前缀和的个数

prefixSumCount.put(currSum, prefixSumCount.getOrDefault(currSum, 0) + 1);
~~~

~~~java
 	public int pathSum(TreeNode root, int sum) {
        // key是前缀和, value是大小为key的前缀和出现的次数
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        // 前缀和为0的一条路径
        prefixSumCount.put(0, 1);
        // 前缀和的递归回溯思路
        return recursionPathSum(root, prefixSumCount, sum, 0);
    }

    /**
     * 前缀和的递归回溯思路
     * 从当前节点反推到根节点(反推比较好理解，正向其实也只有一条)，有且仅有一条路径，因为这是一棵树
     * 如果此前有和为currSum-target,而当前的和又为currSum,两者的差就肯定为target了
     * 所以前缀和对于当前路径来说是唯一的，当前记录的前缀和，在回溯结束，回到本层时去除，保证其不影响其他分支的结果
     * @param node 树节点
     * @param prefixSumCount 前缀和Map
     * @param target 目标值
     * @param currSum 当前路径和
     * @return 满足题意的解
     */
    private int recursionPathSum(TreeNode node, Map<Integer, Integer> prefixSumCount, int target, int currSum) {
        // 1.递归终止条件
        if (node == null) {
            return 0;
        }
        // 2.本层要做的事情
        int res = 0;
        // 当前路径上的和
        currSum += node.val;

        //---核心代码
        // 看看root到当前节点这条路上是否存在节点前缀和加target为currSum的路径
        // 当前节点->root节点反推，有且仅有一条路径，如果此前有和为currSum-target,而当前的和又为currSum,两者的差就肯定为target了
        // currSum-target相当于找路径的起点，起点的sum+target=currSum，当前点到起点的距离就是target
        res += prefixSumCount.getOrDefault(currSum - target, 0);
        // 更新路径上当前节点前缀和的个数
        prefixSumCount.put(currSum, prefixSumCount.getOrDefault(currSum, 0) + 1);
        //---核心代码

        // 3.进入下一层
        res += recursionPathSum(node.left, prefixSumCount, target, currSum);
        res += recursionPathSum(node.right, prefixSumCount, target, currSum);

        // 4.回到本层，恢复状态，去除当前节点的前缀和数量
        prefixSumCount.put(currSum, prefixSumCount.get(currSum) - 1);
        return res;
    }
~~~



#### [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

难度中等434收藏分享切换为英文接收动态反馈

给你一棵 完全二叉树 的根节点 `root` ，求出该树的节点个数。

[完全二叉树](https://baike.baidu.com/item/%E5%AE%8C%E5%85%A8%E4%BA%8C%E5%8F%89%E6%A0%91/7773232?fr=aladdin) 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 `h` 层，则该层包含 $1- 2^h$ 个节点(根节点作为第$0$层)。

 

**示例 1：**

```
输入：root = [1,2,3,4,5,6]
输出：6

```

**示例 2：**

```
输入：root = []
输出：0

```

**示例 3：**

```
输入：root = [1]
输出：1

```

 

**提示：**

- 树中节点的数目范围是`[0, 5 * 104]`
- `0 <= Node.val <= 5 * 104`
- 题目数据保证输入的树是 **完全二叉树**



**进阶：**遍历树来统计节点是一种时间复杂度为 $O(n)$ 的简单解决方案。你可以设计一个更快的算法吗？

通过次数78,731提交次数102,482

**简单递归算法**

~~~java
 	public int countNodes(TreeNode root) {
      return root==null?0:1+countNodes(root.left)+countNodes(root.right);
    }
~~~



**二分查找+位运算方法**

规定根节点位于第 $0$ 层，完全二叉树的最大层数为 $h$。根据完全二叉树的特性可知，完全二叉树的最左边的节点一定位于最底层，因此从根节点出发，每次访问左子节点，直到遇到叶子节点，该叶子节点即为完全二叉树的最左边的节点，经过的路径长度即为最大层数 $h$。

因此对于最大层数为 $h$ 的完全二叉树，节点个数一定在$[2^{h},2^{h+1}-1]$范围内，可以在该范围内通过二分查找的方式得到完全二叉树的节点个数。

具体做法是，根据节点个数范围的上下界得到当前需要判断的节点个数 $k$，如果第 $k$ 个节点存在，则节点个数一定大于或等于 $k$，如果第 $k$ 个节点不存在，则节点个数一定小于 $k$，由此可以将查找的范围缩小一半，直到得到节点个数。

如何判断第 $k$ 个节点是否存在呢？如果第 $k$ 个节点位于第 $h$ 层，则 $k$ 的二进制表示包含 $h+1$ 位，其中最高位是 $1$，其余各位从高到低表示从根节点到第 $k$ 个节点的路径，$0$ 表示移动到左子节点，$1$ 表示移动到右子节点。通过位运算得到第 $k$ 个节点对应的路径，判断该路径对应的节点是否存在，即可判断第 $k$ 个节点是否存在。



![fig1](https://assets.leetcode-cn.com/solution-static/222/1.png)

~~~java
 public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int level = 0;
        TreeNode node = root;
        while (node.left != null) {
            level++;
            node = node.left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (exists(root, level, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public boolean exists(TreeNode root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0) {
            if ((bits & k) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }
~~~

时间复杂度：$O(\log^2{n})$，其中 $n$ 是完全二叉树的节点数。

首先需要 $O(h)$ 的时间得到完全二叉树的最大层数，其中 $h$ 是完全二叉树的最大层数。
使用二分查找确定节点个数时，需要查找的次数为 $O(\log^ {2^{h}})$=$O(h)$，每次查找需要遍历从根节点开始的一条长度为 $h$ 的路径，需要 $O(h)$ 的时间，因此二分查找的总时间复杂度是 $O(h^2)$。由于完全二叉树满足 $2^h \le n < 2^{h+1}$ ，因此有 $O(h)=O(logn)$，$O(h^2)=O(\log^2 n)$。



#### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

难度中等188收藏分享切换为英文接收动态反馈

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

`     3    / \   4   5  / \ 1   2`
给定的树 B：

`   4   / 1`
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

**示例 1：**

```
输入：A = [1,2,3], B = [3,1]
输出：false

```

**示例 2：**

```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```

**限制：**

`0 <= 节点个数 <= 10000`

通过次数69,746提交次数151,877

note:这个比较的是相同子结构，不是要求根节点完全相同

~~~java
 public boolean isSameStructureTree(TreeNode root1, TreeNode root2) {
        if (root2 == null) {
            return true;
        }
        if (root1 == null) {
            return false;
        }
        if (root2.left == null && root2.right == null) {
            return root1.val == root2.val;
        }
        return root1.val == root2.val && isSameStructureTree(root1.left, root2.left) && isSameStructureTree(root1.right, root2.right);
    }

    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) {
            return false;
        }
        return isSameStructureTree(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
~~~



#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

难度中等874收藏分享切换为英文接收动态反馈

根据一棵树的前序遍历与中序遍历构造二叉树。

**注意:**
你可以假设树中没有重复的元素。

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```

返回如下的二叉树：

```
    3
   / \
  9  20
    /  \
   15   7
```

通过次数154,322提交次数223,354

**传统做法**

通过递归的方式，构造根节点以及对应的左子树、右子树，首先从先序遍历中取出根节点，然后去中序遍历中找到对应的根节点，每次在中序遍历中查找根节点耗时为$O(N)$，算法总的时间复杂度是$O(N^2)$。代码如下

~~~java
  public TreeNode constructTree(int[] preorder,int i,int j,int[] inorder,int m,int n){
        if(j-i!=n-m){
            return null;
        }
        if(i>j){
            return null;
        }
        int length=0;
        TreeNode root=new TreeNode(preorder[i]);
        while(inorder[m+length]!=root.val){
            length++;
        }
        TreeNode left=constructTree(preorder,i+1,i+length,inorder,m,m+length-1);
        TreeNode right=constructTree(preorder,i+length+1,j,inorder,m+length+1,n);
        root.left=left;
        root.right=right;   
        return root;
    }
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return constructTree(preorder,0,preorder.length-1,inorder,0,inorder.length-1);
    }
~~~



**优化方法（构造节点以及其在数组中的位置）**

考虑到题目中指出，数中没有重复的节点，我们可以构造一个map对象，用来存储**中序遍历**节点值以及其在数组中的索引位置，这样每次迭代查找根节点的时间复杂度就变成了$O(1)$，总的时间复杂度变成了$O(N)$。

~~~java
	private Map<Integer, Integer> indexMap;

    public TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }

        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = indexMap.get(preorder[preorder_root]);
        
        // 先把根节点建立出来
        TreeNode root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        // 构造哈希映射，帮助我们快速定位根节点
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
~~~



#### [701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

难度中等158收藏分享切换为英文接收动态反馈

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 **保证** ，新值和原始二叉搜索树中的任意节点值都不同。

**注意**，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 **任意有效的结果** 。

 

**示例 1：**

```
输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]
解释：另一个满足题目要求可以通过的树是：


```

**示例 2：**

```
输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]

```

**示例 3：**

```
输入：root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
输出：[4,2,7,1,3,5]
```

 

 

**提示：**

- 给定的树上的节点数介于 `0` 和 `10^4` 之间
- 每个节点都有一个唯一整数值，取值范围从 `0` 到 `10^8`
- `-10^8 <= val <= 10^8`
- 新值和原始二叉搜索树中的任意节点值都不同

通过次数50,199提交次数69,556

**常规做法**

~~~java
 //在二叉搜索树中查找指定的节点，如果该节点存在，那么返回该节点，否则返回最后一个遍历的节点
    public TreeNode searchBST(TreeNode root,TreeNode parent,int target){
        if(root==null){
            return parent;
        }
        if(target==root.val){
            return root;
        }
        if(target>root.val){
            return searchBST(root.right,root,target);
        }else{
            return searchBST(root.left,root,target);
        }
    }
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if(root==null){
            return new TreeNode(val);
        }
        TreeNode lastSearchTarget=searchBST(root,null,val);
        if(val>lastSearchTarget.val){
            lastSearchTarget.right=new TreeNode(val);
        }else{
            lastSearchTarget.left=new TreeNode(val);
        }
        return root;
    }
~~~



**迭代方法**

~~~java
public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        TreeNode pos = root;
        while (pos != null) {
            if (val < pos.val) {
                if (pos.left == null) {
                    pos.left = new TreeNode(val);
                    break;
                } else {
                    pos = pos.left;
                }
            } else {
                if (pos.right == null) {
                    pos.right = new TreeNode(val);
                    break;
                } else {
                    pos = pos.right;
                }
            }
        }
        return root;
    }
~~~



#### [116*. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

难度中等395收藏分享切换为英文接收动态反馈

给定一个 **完美二叉树 **，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。

初始状态下，所有 next 指针都被设置为 `NULL`。

 

**进阶：**

- 你只能使用常量级额外空间。
- 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。



**示例：**

![img](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```
输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。

```

 

**提示：**

- 树中节点的数量少于 `4096`
- `-1000 <= node.val <= 1000`

通过次数98,204提交次数142,775

**我的解法**

~~~java
	public Node connect(Node root) {
        if(root==null){
            return root;
        }
        if(root.left==null){
            return root;
        }
        if(root.left!=null){
            root.left.next=root.right;
        }
        Node next=root.next;
        if(next!=null&&next.left!=null){
            root.right.next=next.left;
        }
        connect(root.left);
        connect(root.right);
        return root;
    }
~~~



**递归解法**

看到题目要求，常数级存储空间，不能采用传统的层次遍历方式来构建next指针

~~~java
	public Node connect(Node root) {
        if(root==null){
            return root;
        }
        if(root.left!=null){
            root.left.next=root.right;
            root.right.next=root.next!=null?root.next.left:null;
            connect(root.left);
            connect(root.right);
        }
        return root;
    }
~~~



**官方版本**

方法二：使用已建立的 $next$ 指针
思路

一棵树中，存在两种类型的$next$ 指针。

第一种情况是连接同一个父节点的两个子节点。它们可以通过同一个节点直接访问到，因此执行下面操作即可完成连接。


$node.left.next = node.right$


第二种情况在不同父亲的子节点之间建立连接，这种情况不能直接连接。



如果每个节点有指向父节点的指针，可以通过该指针找到$next$ 节点。如果不存在该指针，则按照下面思路建立连接：

第 $N$ 层节点之间建立 $next$ 指针后，再建立第 $N+1$ 层节点的 $next$ 指针。可以通过$next$ 指针访问同一层的所有节点，因此可以使用第 $N$ 层的 $next$ 指针，为第$N+1$ 层节点建立 $next$ 指针。

算法

从根节点开始，由于第 $0$ 层只有一个节点，所以不需要连接，直接为第 $1$ 层节点建立 $next$ 指针即可。该算法中需要注意的一点是，当我们为第 $N$ 层节点建立 $next$ 指针时，处于第$N−1$ 层。当第 $N$ 层节点的 $next$ 指针全部建立完成后，移至第 $N$ 层，建立第$N+1$ 层节点的 $next$ 指针。

遍历某一层的节点时，这层节点的 $next$ 指针已经建立。因此我们只需要知道这一层的最左节点，就可以按照链表方式遍历，不需要使用队列。

上面思路的伪代码如下：


leftmost = root
while (leftmost.left != null) {
    head = leftmost
    while (head.next != null) {
        1) Establish Connection 1
        2) Establish Connection 2 using next pointers
        head = head.next
    }
    leftmost = leftmost.left
}


两种类型的 $next$ 指针。

第一种情况两个子节点属于同一个父节点，因此直接通过父节点建立两个子节点的 $next$ 指针即可。


$node.left.next = node.right$


第二种情况是连接不同父节点之间子节点的情况。更具体地说，连接的是第一个父节点的右孩子和第二父节点的左孩子。由于已经在父节点这一层建立了 $next$ 指针，因此可以直接通过第一个父节点的 $next$ 指针找到第二个父节点，然后在它们的孩子之间建立连接。


$node.right.next = node.next.left$


完成当前层的连接后，进入下一层重复操作，直到所有的节点全部连接。进入下一层后需要更新最左节点，然后从新的最左节点开始遍历该层所有节点。因为是完美二叉树，因此最左节点一定是当前层最左节点的左孩子。如果当前最左节点的左孩子不存在，说明已经到达该树的最后一层，完成了所有节点的连接。

~~~java
public Node connect(Node root) {
        if (root == null) {
            return root;
        }
        
        // 从根节点开始
        Node leftmost = root;
        
        while (leftmost.left != null) {
            
            // 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            Node head = leftmost;
            
            while (head != null) {
                
                // CONNECTION 1
                head.left.next = head.right;
                
                // CONNECTION 2
                if (head.next != null) {
                    head.right.next = head.next.left;
                }
                
                // 指针向后移动
                head = head.next;
            }
            
            // 去下一层的最左的节点
            leftmost = leftmost.left;
        }
        
        return root;
    }
~~~



#### [450*. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

难度中等393收藏分享切换为英文接收动态反馈

给定一个二叉搜索树的根节点 $root$ 和一个值 $key$，删除二叉搜索树中的 $key $对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1. 首先找到需要删除的节点；
2. 如果找到了，删除它。

**说明：** 要求算法时间复杂度为 O(h)，h 为树的高度。

**示例:**

```
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7
```

通过次数32,780提交次数71,309

**自己实现复杂版本**

如果删除的节点是叶子节点，直接删除，返回当前节点为空；如果删除的节点不是叶子节点（包含左孩子或者右孩子），需要将孩子节点替换根节点，有两种替换方式，使用左孩子节点替换或者右孩子所在节点替换（两种方式都可以），我实现是采用右孩子节点替换的方式。对于删除的节点包含右孩子，比如树

~~~
			   8
  		  /         \
 		 6		     15
 		/  \		/   \
 	   3    7      10	 18
 	                 \
 	                   13
~~~

我如果要删除根节点$8$,那么我会沿着根节点的右子树$15$所在的节点进行搜索，查找到该子树上最小的节点，然后用来替换根节点。按照二叉搜索树的性质，右字树$15$所在的

最小的节点查找路径沿着$15$的左子树一直遍历下去，直到左子树节点为空，查找路径的最后一个节点就是要替换到根节点的内容，为此需要记录当前遍历的前一个节点以及遍历路径最后一个节点，核心代码如下：

~~~java
 				//pre是p节点的父节点
				TreeNode pre=root;
				//p表示右子树15沿着其左子树遍历的路径的最后一个非空节点
                TreeNode p=root.right;
                //search the min val along root right children
                while(p.left!=null){
                    pre=p;
                    p=p.left;
                }
                //root right children has no left child,replace root with root right children
				//当前根节点的右子树15没有左孩子，直接用当前右子树替换根节点
                if(pre==root){
                    p.left=root.left;
                    root=p;
                }else{
                    //replace root and pRight val,and  pre left point p right children
                    //当前根节点的右子树15有左孩子，那么用该节点的内容替换根节点，同时将父节点pre的右孩子指向该节点的右孩子
                    root.val=p.val;
                    pre.left=p.right;
                }
~~~



同理，左孩子节点和右孩子的处理逻辑类似，完整代码如下：(递归返回的是删除后的根节点)

~~~java
 public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else {
            //leaf node ,delete directly
            if (root.left == null && root.right == null) {
                return null;
            }
            //exist right children, extract right children to replace root val
            if (root.right != null) {
                TreeNode pre=root;
                TreeNode p=root.right;
                //search the min val along root right children
                while(p.left!=null){
                    pre=p;
                    p=p.left;
                }
                //root right children has no left child,replace root with root right children
                if(pre==root){
                    p.left=root.left;
                    root=p;
                }else{
                    //replace root and pRight val,and  pre left point p right children
                    root.val=p.val;
                    pre.left=p.right;
                }
            } else {
                //root has no right children
                //search along root left children to find max val
                TreeNode pre=root;
                TreeNode p=root.left;
                while(p.right!=null){
                    pre=p;
                    p=p.right;
                }
                if(pre==root){
                    //root left children has not right child,replace root with it's left children
                    p.right=root.right;
                    root=p;
                }else{
                    //replace root and p val,and change pre right point p left chilren
                    root.val=p.val;
                    pre.right=p.left;
                }
            }
        }
        return root;
    }
~~~



**网上简单版本**

遍历右孩子节点的时候不用做节点替换操作，如果存在右子树，那么沿着右子树的左孩子一直遍历下去，遍历到最后一个左孩子节点$p$,然后$p$的左孩子指向要删除节点的

左孩子节点即可；如果没有右孩子那么直接返回左子树。**两种不同的删除策略，实际上是相同的效果。**

~~~java
 public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return root;
        }
        if (key < root.val) {
            TreeNode left = deleteNode(root.left, key);
            root.left = left;
        } else if (key > root.val) {
            TreeNode right = deleteNode(root.right, key);
            root.right = right;
        } else {
            TreeNode left = root.left;
            TreeNode right = root.right;
            //寻找右侧最小的叶子节点
            while (right != null && right.left != null) {
                right = right.left;
            }
            //如果存在右侧最小的叶子节点，将root的左子树拼接到右侧最小叶子节点的左子树
            if (right != null) {
                right.left = left;
                return root.right;
            } else {//如果不存在右侧最小的叶子节点，root的右子树为空，直接返回左子树
                return left;
            }
        }
        return root;
    }
}
~~~



#### [889*. 根据前序和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

难度中等141收藏分享切换为英文接收动态反馈

返回与给定的前序和后序遍历匹配的任何二叉树。

 `pre` 和 `post` 遍历中的值是不同的正整数。

 

**示例：**

```
输入：pre = [1,2,4,5,3,6,7], post = [4,5,2,6,7,3,1]
输出：[1,2,3,4,5,6,7]

```

 

**提示：**

- `1 <= pre.length == post.length <= 30`
- `pre[]` 和 `post[]` 都是 `1, 2, ..., pre.length` 的排列
- 每个输入保证至少有一个答案。如果有多个答案，可以返回其中一个。

通过次数9,943提交次数14,768

**我的代码实现版本**

按照基本递归思想，先序和后续遍历由于无法区分左子树和右子树的边界，统一将先序遍历紧邻根节点左侧的看成是左子树。

代码如下：

~~~java
Map<Integer, Integer> preMap = new HashMap();
    Map<Integer, Integer> postMap = new HashMap();

    public TreeNode constructFromPrePost(int[] pre, int i, int j, int[] post, int m, int n) {
        if (i > j) {
            return null;
        }
        if (i == j) {
            return new TreeNode(pre[i]);
        }
        TreeNode root = new TreeNode(pre[i]);
      //该根节点的下一个左孩子节点对应的数组范围
        int leftChildrenLength = postMap.get(pre[i + 1]) - m+1;
        root.left = constructFromPrePost(pre, i + 1, i + leftChildrenLength, post, m, m + leftChildrenLength - 1);
        root.right = constructFromPrePost(pre, i + leftChildrenLength + 1, j, post, m + leftChildrenLength, n - 1);
        return root;
    }

    public TreeNode constructFromPrePost(int[] pre, int[] post) {
        if (pre.length != post.length || pre.length == 0) {
            return null;
        }
        for (int i = 0; i < pre.length; i++) {
            preMap.put(pre[i], i);
            postMap.put(post[i], i);
        }
        return constructFromPrePost(pre, 0, pre.length - 1, post, 0, post.length - 1);
    }
~~~



#### [549.* 二叉树中最长的连续序列](https://leetcode-cn.com/problems/binary-tree-longest-consecutive-sequence-ii/)

难度中等65收藏分享切换为英文接收动态反馈

给定一个二叉树，你需要找出二叉树中最长的连续序列路径的长度。

请注意，该路径可以是递增的或者是递减。例如，[1,2,3,4] 和 [4,3,2,1] 都被认为是合法的，而路径 [1,2,4,3] 则不合法。另一方面，路径可以是 子-父-子 顺序，并不一定是 父-子 顺序。

**示例 1:**

```
输入:
        1
       / \
      2   3
输出: 2
解释: 最长的连续路径是 [1, 2] 或者 [2, 1]。
```

 

**示例 2:**

```
输入:
        2
       / \
      1   3
输出: 3
解释: 最长的连续路径是 [1, 2, 3] 或者 [3, 2, 1]。
```

 

**注意:** 树上所有节点的值都在$ [-e^7,e ^7]$ 范围内。



**我的解法**

最长连续序列容易想到一种递归方法，以当前根节点作为起始节点，最长的递增或者递减的序列长度。由于题目中要求可能是递增的也可能是递减的，为了减少代码实现的复杂度，调用两次，分别表示最长递增的子序列以及最长递减的子序列。以最长递减子序列为例，根节点最长的序列可能是来自于当前根节点和左子树、根节点和右子树，以及根节点和左右子树。(**左右子树这种方式，二者的序列增长方式相反，以递减序列为例，要求从根节点出发的左子树是递减的，从根节点出发的右子树是递增的**)。为了减少重复的查询，采用备忘录模式，记录每次遍历的节点递增或者递减的序列最大长度。代码如下：

~~~java
 	int maxConsecutive = 0;
//	val中的int[]是一个两个元素的数组，第一个表示以当前节点作为根节点递增的最长子序列的长度即num[0],第二个表示以当前节点作为跟几点的递减的最长的子序列的长度num[1]
    Map<TreeNode, int[]> treeNodeMap = new HashMap<>();

    //return longest consecutive path with root as start point,increase indicate increase or decrease
    int longestConsecutive(TreeNode root, boolean increase) {
        if (root == null) {
            return 0;
        }
        int[] depth = treeNodeMap.get(root);
        if (depth == null) {
            depth = new int[2];
            Arrays.fill(depth, -1);
            treeNodeMap.put(root, depth);
        }
        int index = increase ? 0 : 1;
        if (depth[index] != -1) {
            return depth[index];
        }
        if (root.left == null && root.right == null) {
            depth[index]=1;
            maxConsecutive=Math.max(depth[index],maxConsecutive);
            return 1;
        }
        int leftLength = longestConsecutive(root.left, increase);
        int rightLength = longestConsecutive(root.right, increase);
        int inverseRightLength = longestConsecutive(root.right, !increase);
        int rootLength = 1;
        int diff = increase ? 1 : -1;
        if (root.left != null && root.left.val - root.val == diff) {
            rootLength = leftLength + 1;
        }
        if (root.right != null && root.right.val - root.val == diff) {
            rootLength = Math.max(rootLength, rightLength + 1);
        }
        maxConsecutive = Math.max(maxConsecutive, rootLength);
        boolean satisfy = root.left != null && root.right != null && root.left.val - root.val == diff && root.val - root.right.val == diff;
        if (satisfy) {
            maxConsecutive = Math.max(leftLength + inverseRightLength + 1, maxConsecutive);
        }
        depth[index] = rootLength;
        return rootLength;
    }

    public int longestConsecutive(TreeNode root) {
        longestConsecutive(root, true);
        longestConsecutive(root, false);
        return maxConsecutive;
    }
~~~



**官方版本**

~~~java
	int maxval = 0;
    public int longestConsecutive(TreeNode root) {
        longestPath(root);
        return maxval;
    }
    public int[] longestPath(TreeNode root) {
        if (root == null)
            return new int[] {0,0};
        int inr = 1, dcr = 1;
        if (root.left != null) {
            int[] l = longestPath(root.left);
            if (root.val == root.left.val + 1)
                dcr = l[1] + 1;
            else if (root.val == root.left.val - 1)
                inr = l[0] + 1;
        }
        if (root.right != null) {
            int[] r = longestPath(root.right);
            if (root.val == root.right.val + 1)
                dcr = Math.max(dcr, r[1] + 1);
            else if (root.val == root.right.val - 1)
                inr = Math.max(inr, r[0] + 1);
        }
        maxval = Math.max(maxval, dcr + inr - 1);
        return new int[] {inr, dcr};
    }
~~~



#### [669*. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

难度中等350收藏分享切换为英文接收动态反馈

给你二叉搜索树的根节点 `root` ，同时给定最小边界`low` 和最大边界 `high`。通过修剪二叉搜索树，使得所有节点的值在`[low, high]`中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

 

**示例 1：**

```
输入：root = [1,0,2], low = 1, high = 2
输出：[1,null,2]

```

**示例 2：**

```
输入：root = [3,0,4,null,2,null,null,1], low = 1, high = 3
输出：[3,2,null,1]

```

**示例 3：**

```
输入：root = [1], low = 1, high = 2
输出：[1]

```

**示例 4：**

```
输入：root = [1,null,2], low = 1, high = 3
输出：[1,null,2]

```

**示例 5：**

```
输入：root = [1,null,2], low = 2, high = 4
输出：[2]

```

 

**提示：**

- 树中节点数在范围 `[1, 104]` 内
- `0 <= Node.val <= 104`
- 树中每个节点的值都是唯一的
- 题目数据保证输入是一棵有效的二叉搜索树
- `0 <= low <= high <= 104`

**网上代码**

第一次想错了，认为就是简单的如果节点不满足给定的范围，那么就删除该节点。实际上根据二叉搜索树的性质，比较简单的递归就能满足要求。如果当前根节点小于最小值，

那么该节点的左孩子也全部作废，直接返回剪裁后的右子树。代码如下：

~~~java
public TreeNode trimBST(TreeNode root, int low, int high) {
       if(root==null){
           return null;
       }
       //当前节点小于最小值，那么左孩子所有节点也小于该节点
       if(root.val<low){
           return trimBST(root.right,low,high);
       }
       if(root.val>high){
           return trimBST(root.left,low,high);
       }
       //
       root.left=trimBST(root.left,low,high);
       root.right=trimBST(root.right,low,high);
       return root;
    }
~~~



#### [285.* 二叉搜索树中的顺序后继](https://leetcode-cn.com/problems/inorder-successor-in-bst/)

难度中等88收藏分享切换为英文接收动态反馈

给你一个二叉搜索树和其中的某一个结点，请你找出该结点在树中顺序后继的节点。

结点 `p` 的后继是值比 `p.val` 大的结点中键值最小的结点。

 

**示例 1:**

![img](https://assets.leetcode.com/uploads/2019/01/23/285_example_1.PNG)

```
输入: root = [2,1,3], p = 1
输出: 2
解析: 这里 1 的顺序后继是 2。请注意 p 和返回值都应是 TreeNode 类型。

```

**示例 2:**

![img](https://assets.leetcode.com/uploads/2019/01/23/285_example_2.PNG)

```
输入: root = [5,3,6,2,4,null,null,1], p = 6
输出: null
解析: 因为给出的结点没有顺序后继，所以答案就返回 null 了。

```

 

**注意:**

1. 假如给出的结点在该树中没有顺序后继的话，请返回 `null`
2. 我们保证树中每个结点的值是唯一的

通过次数5,431提交次数8,581

**答案**

常规做法，利用二叉搜索树的性质，设当前所在根节点是$root$，递归结束条件是当前节点为空或者是叶子；否则遍历左右子树，如果目标值不小于$root$,后继节点肯定在右子树上（也有可能为空），**否则一定在左子树或者是当前根节点**（如果左子树上存在其后继，那么是当前节点的最近后继，否则就是当前根节点）。

~~~java
public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if(root==null){
            return null;
        }
        if(root.left==null&&root.right==null){
            if(root.val>p.val){
                return root;
            }
            return null;
        }
        if(p.val>=root.val){
            return inorderSuccessor(root.right, p);
        }else{
            TreeNode leftChild= inorderSuccessor(root.left, p);
            if(leftChild!=null){
                return leftChild;
            }
            return root;
        }
    }
~~~



**非递归做法**

节点$c$表示不小于$p$的最后一个节点

~~~java
public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode r = null;
        
        TreeNode c = root;
        
        while(c!=null && c!=p){
            if(p.val < c.val){
                r = c;
                c = c.left;
            } else {
                c = c.right;
            }
        }
        if(c == null || c.right == null) return r;
        //当前节点c==p并且c.right!=null
        c = c.right;
        
        while(c.left!= null){
            c = c.left;
        }
        
        return c;
    }
~~~



#### [255*. 验证前序遍历序列二叉搜索树](https://leetcode-cn.com/problems/verify-preorder-sequence-in-binary-search-tree/)

难度中等76收藏分享切换为英文接收动态反馈

给定一个整数数组，你需要验证它是否是一个二叉搜索树正确的先序遍历序列。

你可以假定该序列中的数都是不相同的。

参考以下这颗二叉搜索树：

```
     5
    / \
   2   6
  / \
 1   3
```

**示例 1：**

```
输入: [5,2,6,1,3]
输出: false
```

**示例 2：**

```
输入: [5,2,1,3,6]
输出: true
```

**进阶挑战：**

您能否使用恒定的空间复杂度来完成此题？

通过次数3,593提交次数7,809

**题解**

刚开始没有思路，看了评论才知道如何去判断。先序遍历顺序是$root>left>right$的形式，也就是说从当前遍历节点出发，遍历整个序列查找到第一个比当前节点大的节点，那么该节点以及该节点之后的序列都是右子树，即要求该节点以及以后的节点的值都大于根节点。

双重循环暴力破解，算法时间复杂度$O(n^2)$

~~~

~~~



**方法二单调栈**

维护一个单调递减栈（从栈顶到栈底），如某一状态下栈元素为$[5,4,3]$。若碰到一个$6$的时候，说明从左子树（或者没有左子树）到达了右子树，此时将小于$6$的元素都pop掉，栈变成$[6]$,并且记录一个最小值为$5$，由于$6$是右子树，因此$6$右侧的元素都必须大于$5$，否则不合法。

~~~java
// 用单调栈的方式，递减栈，当碰到一个数比栈顶元素大的时候，说明从左子树到了右子树。
    // 此时要删掉左子树的所有节点，并且保留子树的根为最小值，此时遍历的所有右子树的节点都必须大于这个根，否则非法
    public boolean verifyPreorder(int[] preorder) {
        int len = preorder.length;
        int[] stack = new int[len];
        int top = -1;
        int min = Integer.MIN_VALUE;

        for (int value : preorder) {
            if (value < min) {
                return false;
            }

            while (top > -1 && value > stack[top]) {
                min = stack[top];
                top--;
            }

            stack[++top] = value;
        }

        return true;
    }
~~~



#### [117*. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

难度中等363收藏分享切换为英文接收动态反馈

给定一个二叉树

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。

初始状态下，所有 next 指针都被设置为 `NULL`。

 

**进阶：**

- 你只能使用常量级额外空间。
- 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。



**示例：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/15/117_sample.png)

```
输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。
```

 

**提示：**

- 树中的节点数小于 `6000`
- `-100 <= node.val <= 100`





通过次数63,278提交次数106,434

**空间复杂度为$O(n)$的解法**

比较简单，不说了

~~~java
public Node connect(Node root) {
        Queue<Node> queue=new LinkedList();
        if(root==null){
            return root;
        }
        queue.offer(root);
        while(!queue.isEmpty()){
            int size=queue.size();
            Node pre=null;
            for(int i=0;i<size;i++){
                Node p=queue.poll();
                if(pre!=null){
                    pre.next=p;
                }
                pre=p;
                if(p.left!=null){
                    queue.offer(p.left);
                }
                if(p.right!=null){
                    queue.offer(p.right);
                }
            }
        }
        return root;
    }
~~~



**DFS，空间复杂度是$o(h)$**

虽然递归无法保证按照层次的顺序进行遍历，但是可以确保相同层次的节点遍历顺序是从左到右的，先序遍历。使用map记录每一层最新的遍历的节点

~~~java
 Map<Integer, Node> map = new HashMap<>();
    public Node connect(Node root) {
        helper(root, 0);
        return root;
    }

    void helper(Node node, int deepth){
        if(node == null) return;
        if(map.containsKey(deepth)){
            map.get(deepth).next = node;
        }
        map.put(deepth, node);
        helper(node.left, deepth + 1);
        helper(node.right, deepth + 1);
    }
~~~



**官方题解，空间复杂度为O(1)**

~~~java
public Node connect(Node root) {
        if (root == null)
            return root;
        //cur我们可以把它看做是每一层的链表
        Node cur = root;
        while (cur != null) {
            //遍历当前层的时候，为了方便操作在下一
            //层前面添加一个哑结点（注意这里是访问
            //当前层的节点，然后把下一层的节点串起来）
           //dummpy和pre指的是下一层的节点，利用当前层的next指针顺序来构建下一层的顺序
            Node dummy = new Node(0);
            //pre表示访下一层节点的前一个节点
            Node pre = dummy;
            //然后开始遍历当前层的链表
            while (cur != null) {
                if (cur.left != null) {
                    //如果当前节点的左子节点不为空，就让pre节点
                    //的next指向他，也就是把它串起来
                    pre.next = cur.left;
                    //然后再更新pre
                    pre = pre.next;
                }
                //同理参照左子树
                if (cur.right != null) {
                    pre.next = cur.right;
                    pre = pre.next;
                }
                //继续访问这样行的下一个节点
                cur = cur.next;
            }
            //把下一层串联成一个链表之后，让他赋值给cur，
            //后续继续循环，直到cur为空为止
            cur = dummy.next;
        }
        return root;
    }
~~~



#### [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

难度困难473收藏分享切换为英文接收动态反馈

序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

**提示: **输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 [LeetCode 序列化二叉树的格式](https://leetcode-cn.com/faq/#binary-tree)。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

 

**示例 1：**

```
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]

```

**示例 2：**

```
输入：root = []
输出：[]

```

**示例 3：**

```
输入：root = [1]
输出：[1]

```

**示例 4：**

```
输入：root = [1,2]
输出：[1,2]

```

 

**提示：**

- 树中结点数在范围 `[0, 104]` 内
- `-1000 <= Node.val <= 1000`

通过次数67,557提交次数125,596

**开发性答案**

我的解答步骤，如何表示一个节点，我是将一个节点表示成如下形式，val@parentIndex@childFlag，其中val表示当前节点的值，parentIndex表示父节点所在的层次遍历的索引（索引从0开始），childFlag表示当前节点是左节点还是右节点。

~~~java
public class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            //each node consist with val@parentIndex@(L|R), root node has no parent index ,set -1, L OR R indicate left or right children
            //level traversal
            StringBuffer result = new StringBuffer();
            if (root == null) {
                return result.toString();
            }
            Queue<TreeNode> queue = new LinkedList();
            queue.offer(root);
            int index = 0;
            Map<TreeNode, Integer> treeNode2LevelIndexMap = new LinkedHashMap<>();
            Map<TreeNode, TreeNode> child2ParentMap = new HashMap();
            Map<TreeNode, String> child2LOrRMap = new HashMap<>();
            while (!queue.isEmpty()) {
                int size = queue.size();
                for (int i = 0; i < size; i++) {
                    TreeNode p = queue.poll();
                    treeNode2LevelIndexMap.put(p, index++);
                    if (p.left != null) {
                        queue.offer(p.left);
                        child2ParentMap.put(p.left, p);
                        child2LOrRMap.put(p.left, "L");
                    }
                    if (p.right != null) {
                        queue.offer(p.right);
                        child2ParentMap.put(p.right, p);
                        child2LOrRMap.put(p.right, "R");
                    }
                }
            }
            //construct node unit
            for (Map.Entry<TreeNode, Integer> entry : treeNode2LevelIndexMap.entrySet()) {
                TreeNode treeNode = entry.getKey();
                TreeNode parent = child2ParentMap.get(treeNode);
                result.append(treeNode.val + "@");
                if (parent == null) {
                    //it's tree root node
                    result.append("-1");
                } else {
                    result.append(treeNode2LevelIndexMap.get(parent)).append("@").append(child2LOrRMap.get(treeNode));
                }
                result.append(",");
            }
            //remove last comma
            if (result.length() > 0) {
                result.deleteCharAt(result.length() - 1);
            }
            return result.toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.trim().isEmpty()) {
                return null;
            }
            String[] nodeValStrArr = data.split(",");
            Map<Integer, TreeNode> levelIndex2TreeNodeMap = new HashMap<>();
            int index = 0;
            TreeNode root = null;
            for (String item : nodeValStrArr) {
                String[] unit = item.split("@");
                TreeNode node = new TreeNode(Integer.parseInt(unit[0]));
                levelIndex2TreeNodeMap.put(index++, node);
                int parentIndex = Integer.parseInt(unit[1]);
                if (parentIndex == -1) {
                    root = node;
                } else {
                    String childFlag = unit[2];
                    TreeNode parent = levelIndex2TreeNodeMap.get(parentIndex);
                    if (childFlag.equals("L")) {
                        parent.left = node;
                    } else {
                        parent.right = node;
                    }
                }

            }
            return root;
        }
    }
~~~



**网上优秀解答**

直接利用树的先序遍历，借助一个临时存储，记录先序遍历的访问顺序，在反序列化的时候，按照递归遍历的顺序进行反序列化。

~~~java
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
      //这个标记为用来区分左孩子还是右孩子
        if (root == NULL)
            return "#_";
        string res = to_string(root->val) + "_";
        res += serialize(root->left);
        res += serialize(root->right);
        return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        std::stringstream ss(data);
        std::string item;
        queue<string> q;
        while (std::getline(ss, item, '_')) 
            q.push(item);
        return helper(q);
    }
    TreeNode* helper(queue<string>& q)
    {
        string val = q.front();
        q.pop();
        if (val == "#")
            return NULL;
        TreeNode* head = new TreeNode(stoi(val));
        head->left = helper(q);
        head->right = helper(q);
        return head;
    }
};
~~~



#### [666. 路径总和 IV](https://leetcode-cn.com/problems/path-sum-iv/)

难度中等26收藏分享切换为英文接收动态反馈

对于一棵深度小于 `5` 的树，可以用一组三位十进制整数来表示。

对于每个整数：

1. 百位上的数字表示这个节点的深度 `D`，`1 <= D <= 4`。
2. 十位上的数字表示这个节点在当前层所在的位置 `P`， `1 <= P <= 8`。位置编号与一棵满二叉树的位置编号相同。
3. 个位上的数字表示这个节点的权值 `V`，`0 <= V <= 9`。

给定一个包含三位整数的`升序`数组，表示一棵深度小于 5 的二叉树，请你返回从根到所有叶子结点的路径之和。

 

**示例 1：**

```
输入: [113, 215, 221]
输出: 12
解释: 
这棵树形状如下:
    3
   / \
  5   1

路径和 = (3 + 5) + (3 + 1) = 12.

```

**示例 2：**

```
输入: [113, 221]
输出: 4
解释: 
这棵树形状如下: 
    3
     \
      1

路径和 = (3 + 1) = 4.

```

 

通过次数1,528提交次数2,474

**解答**

常规做法，通过输入的数组来构建树的结构，由于最后一位个位数是按照满二叉树的所在层次的编号进行编排的，自然想到通过构建满二叉树的索引来构建这棵树，由于索引是从$0$开始的，对于满二叉树而言，假设根节点所在的索引是$index$,那么左右孩子节点的索引分别是$2*index+1$和$2*index+2$，如果一个节点的左右孩子索引都不存在，自然这个节点就是叶子节点。构建满二叉树的索引结构核心代码如下：(level当前层次，levelPos满二叉树的当前层序号)

~~~java
 for (int i = 0; i < nums.length; i++) {
            int level = nums[i] / 100;
            int levelPos = nums[i] / 10 % 10;
            int val = nums[i] % 10;
            int realPos = (int) (Math.pow(2, level - 1) + levelPos-1);
            index2ValMap.put(realPos - 1, val);
        }
~~~



完整代码:

~~~java
int sumTotal = 0;
    void preOrder(Map<Integer, Integer> index2ValMap, int rootIndex, int pathSum) {
        if (!index2ValMap.containsKey(rootIndex)) {
            return;
        }
        int leftChildIndex = 2 * rootIndex + 1;
        int rightChildIndex = 2 * rootIndex + 2;
        pathSum += index2ValMap.get(rootIndex);
        //leaf node
        if (!index2ValMap.containsKey(leftChildIndex) && !index2ValMap.containsKey(rightChildIndex)) {
            sumTotal += pathSum;
            return;
        }
        preOrder(index2ValMap, leftChildIndex, pathSum);
        preOrder(index2ValMap, rightChildIndex, pathSum);
    }

    public int pathSum(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        Map<Integer, Integer> index2ValMap = new HashMap();
        for (int i = 0; i < nums.length; i++) {
            int level = nums[i] / 100;
            int levelPos = nums[i] / 10 % 10;
            int val = nums[i] % 10;
            int realPos = (int) (Math.pow(2, level - 1) + levelPos-1);
            index2ValMap.put(realPos - 1, val);
        }
        preOrder(index2ValMap, 0, 0);
        return sumTotal;
    }
~~~



#### [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

难度困难423收藏分享切换为英文接收动态反馈

给你二叉搜索树的根节点 `root` ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

**进阶：**使用 O(*n*) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

 

**示例 1：**

```
输入：root = [1,3,null,null,2]
输出：[3,1,null,null,2]
解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。

```

**示例 2：**

```
输入：root = [3,1,4,null,null,2]
输出：[2,1,4,null,null,3]
解释：2 不能在 3 的右子树中，因为 2 < 3 。交换 2 和 3 使二叉搜索树有效。
```

 

**提示：**

- 树上节点的数目在范围 `[2, 1000]` 内
- `-231 <= Node.val <= 231 - 1`

通过次数48,800提交次数78,581

**O(n)复杂度的算法实现**

代码里面加了比较详细的注释，直接看代码即可

~~~java
  //O(n)的可以尝试，log(n)的算了
    //对于异常的二叉搜索树，最多有两种相邻的逆序，比如1,2,3,4,5
    //一种可能逆序是1,2,4,3,5那么直接交换4和3的顺序就行了（这个是相邻节点的逆序）；
    //还有一种逆序结构是不相邻的节点的逆序，比如2和5交换，结构是4,5,1,3,2，逆序结构是5，1和3,2，交换5和1就可以了
    //综上，可以采用记录第一个初始逆序节点和最后一个逆序节点，然后进行交换
    //结构如下：
    //初始   3            相邻逆序结构        4         不相邻逆序结构              4
    //    2     4                       2      3                              5     3
    //  1           5                 1            5                        1           2


    TreeNode t1,t2,pre;
    void inOrder(TreeNode root){
        if(root!=null){
            inOrder((root.left));
            if(pre!=null){
                //change order
                if(pre.val>=root.val){
                    //取初始的第一个逆序的最左边，即pre节点                  
                   if(t1==null){
                       t1=pre;
                   }
                   //取最后一个逆序的最右边，即root节点
                   t2=root;
                }
            }
            pre=root;
            inOrder(root.right);
        }
    }
    public void recoverTree(TreeNode root) {
       inOrder(root);
       int tmp=t1.val;
       t1.val=t2.val;
       t2.val=tmp;
    }
~~~



**方法morris中序遍历（看不懂）**

~~~java
  public void recoverTree(TreeNode root) {
        TreeNode x = null, y = null, pred = null, predecessor = null;

        while (root != null) {
            if (root.left != null) {
                // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                predecessor = root.left;
                while (predecessor.right != null && predecessor.right != root) {
                    predecessor = predecessor.right;
                }
                
                // 让 predecessor 的右指针指向 root，继续遍历左子树
                if (predecessor.right == null) {
                    predecessor.right = root;
                    root = root.left;
                }
                // 说明左子树已经访问完了，我们需要断开链接
                else {
                    if (pred != null && root.val < pred.val) {
                        y = root;
                        if (x == null) {
                            x = pred;
                        }
                    }
                    pred = root;

                    predecessor.right = null;
                    root = root.right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                if (pred != null && root.val < pred.val) {
                    y = root;
                    if (x == null) {
                        x = pred;
                    }
                }
                pred = root;
                root = root.right;
            }
        }
        swap(x, y);
    }

    public void swap(TreeNode x, TreeNode y) {
        int tmp = x.val;
        x.val = y.val;
        y.val = tmp;
    }

~~~



#### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

难度中等733收藏分享切换为英文接收动态反馈

**动态规划方法**

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

**示例 1:**

```
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
```

**示例 2:**

```
输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.

```

通过次数86,141提交次数140,636

由题目要求不能看出，容易推理出动态规划的方法，指的$opt(i)$是以节点$i$作为父节点的能够盗取的最高金额，那么存在二分方法。小偷盗取当前节点所在房子和小偷不盗取当前节点所在房子的金额。采用map来做备忘录，详细代码如下：

~~~java
 //动态规划方式实现，opt(i)表示以当前节点作为父节点出发最大打劫金额，容易通过推理得到
    //opt(i)=Math.max[opt(i.left)+opt(i.right),
    //node[i]+opt(i.left.left)+opt(i.left.right)+opt(i.right.left)+ opt(i.right.right));
    Map<TreeNode,Integer> node2RobMap=new HashMap();
    public int rob(TreeNode root) {
        if(root==null){
            return 0;
        }
        if(node2RobMap.containsKey(root)){
            return node2RobMap.get(root);
        }
        //不选择打劫当前root节点的房间，那么最大值是左右房间打劫金额总和
        int maxVal=rob(root.left)+rob(root.right);
        //选择打劫当前root节点的房间
        int chooseNode=root.val;
        if(root.left!=null){
            chooseNode+=rob(root.left.left)+rob(root.left.right);
        }
        if(root.right!=null){
            chooseNode+=rob(root.right.left)+rob(root.right.right);
        }
        maxVal= Math.max(maxVal,chooseNode);
        node2RobMap.put(root,maxVal);
        return maxVal;
    }
~~~



**网上精简代码**

~~~java
public int rob(TreeNode root) {
        int[] res=dp(root);
        return Math.max(res[0],res[1]);
    }
    //递归方法中返回的是小偷选择当前节点所能盗取的最大金额和小偷不选择当前节点所能盗取的最大金额
	//这种方式不需要记录每个节点出发所能盗取的最大金额
    int[] dp(TreeNode root){
        if(root==null){
            return new int[]{0,0};
        }

        int[] left=dp(root.left);
        int[] right=dp(root.right);
		
        int do_it=root.val+left[0]+right[0];
        int not_do=Math.max(left[0],left[1])+Math.max(right[0],right[1]);

        return new int[]{not_do,do_it};
    }
~~~



### 2021年3月

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

难度中等5703收藏分享切换为英文接收动态反馈

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

 

**示例 1：**

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

```

**示例 2：**

```
输入：l1 = [0], l2 = [0]
输出：[0]

```

**示例 3：**

```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

```

 

**提示：**

- 每个链表中的节点数在范围 `[1, 100]` 内
- `0 <= Node.val <= 9`
- 题目数据保证列表表示的数字不含前导零

通过次数723,525提交次数1,826,704

**我的解答**

根据题目要求，数字的存储顺序是按照链表访问顺序逆序排列的——即低位放在前面，高位放在后面，这种方式其实是一种自然的运算顺序，提取出对应的值并进行累加。

注意进位的影响。

~~~java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1==null){
            return l2;
        }
        if(l2==null){
            return l1;
        }
        //store final result,from low bit to high
        ListNode p1=l1,p2=l2;
        //final return result,store tail insert mode
        ListNode head=null,p=null;
        //indicate carry flag
        int carry=0;
        while(p1!=null||p2!=null){
            int sum=carry;
            if(p1!=null){
                sum+=p1.val;
                p1=p1.next;
            }
            if(p2!=null){
                sum+=p2.val;
                p2=p2.next;
            }
            carry=sum>9?1:0;
            ListNode node=new ListNode(sum%10);
            if(head==null){
                head=node;
                p=head;
            }else{
                p.next=node;
                p=node;
            }
        }
        if(carry>0){
            p.next=new ListNode(carry);
        }
        return head;  
    }
~~~



#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

难度中等3261收藏分享切换为英文接收动态反馈

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

 

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"

```

**示例 3：**

```
输入：s = "a"
输出："a"

```

**示例 4：**

```
输入：s = "ac"
输出："a"

```

 

**提示：**

- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母（大写和/或小写）组成

通过次数489,902提交次数1,473,938

**传统暴力破解法**

~~~java
 public boolean match(int left,int right,String s){
        int length=s.length();
        if(left<0||right>length-1){
            return false;
        }
        return s.charAt(left)==s.charAt(right);
    }
    public String longestPalindrome(String s) {
        //暴力破解法，算法复杂度是O（n^2）
        //回文串从结构上有两种形式，一种是奇数串（以中间节点做分隔），另一种是偶数串，以左右两边做对比
        int maxResult=0,maxLeft=0,maxRight=0;
        if(s==null){
            return s;
        }
        for(int i=0;i<s.length();i++){
            //奇数
            int odd=1;
            int j=1;
            while(match(i-j,i+j,s)){
                odd+=2;
                j++;
            }
            if(odd>maxResult){
                maxLeft=i-j+1;
                maxRight=i+j-1;
                maxResult=odd;
            }
            int even=0;
            j=0;
            while(match(i-j,i+j+1,s)){
                even+=2;
                j++;
            }
            if(even>maxResult){
                maxLeft=i-j+1;
                maxRight=i+j;
                maxResult=even;
            }
        }
        return s.substring(maxLeft,maxRight+1);
    }
~~~



**动态规划方法**

动态规划转移方程

$$
	P(i,j)=P(i+1,j−1)∧(Si==Sj)
$$

注意上述的动态规划转移方程，要求$P(i,j)$的长度不大于2，边界条件是

$$
P(i,i)=true\\
	P(i,i+1)=(Si==Si+1)
$$


**注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。**

~~~java
public String longestPalindrome(String s) {
        int n = s.length();	
        boolean[][] dp = new boolean[n][n];
        String ans = "";
  		//注意动态规划最外层的转移条件
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = true;
                } else if (l == 1) {
                    dp[i][j] = (s.charAt(i) == s.charAt(j));
                } else {
                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.length()) {
                    ans = s.substring(i, i + l + 1);
                }
            }
        }
        return ans;
    }
~~~



#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

难度中等5041收藏分享切换为英文接收动态反馈

给定一个字符串，请你找出其中不含有重复字符的 **最长子串 **的长度。

 

**示例 1:**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

```

**示例 2:**

```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

```

**示例 3:**

```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

```

**示例 4:**

```
输入: s = ""
输出: 0

```

 

**提示：**

- `0 <= s.length <= 5 * 104`
- `s` 由英文字母、数字、符号和空格组成

通过次数849,767提交次数2,326,466

传统的暴力破解法

算法时间复杂度是$o(n^2)$

~~~java
public int lengthOfLongestSubstring(String s) {
        //暴力破解法，算法时间复杂度是O(n^2)
        if(s==null||s.length()==0){
            return 0;
        }
        int maxLength=1;
        for(int i=0;i+maxLength-1<s.length();i++){
            Set<Character> set=new HashSet();
            set.add(s.charAt(i));
            int j=i+1;
            for(;j<s.length();j++){
                if(set.contains(s.charAt(j))){
                    break;
                }
                set.add(s.charAt(j));
            }
            //在最外层判断
            maxLength=Math.max(maxLength,j-i);
        }
        return maxLength;
    }
~~~



**优化算法1**

用两个map分别记录已经访问过的字符序列，每个字符最多出入一次，理论上算法时间复杂度是$O(n)$，但是因为map的移动算法实际复杂度会高于该值。

如果使用$LinkedHashMap$实现上会更简单一些

~~~java
public int lengthOfLongestSubstring(String s) {
        //暴力破解法，算法时间复杂度是O(n^2)
        if(s==null||s.length()==0){
            return 0;
        }
        int maxLength=1,left=0,right=0;
        Map<Integer,Character> pos2ChMap=new HashMap();
        Map<Character,Integer> ch2PosMap=new HashMap();
        for(;left+maxLength-1<s.length();){
            while(right<s.length()&&!ch2PosMap.containsKey(s.charAt(right))){
                pos2ChMap.put(right,s.charAt(right));
                ch2PosMap.put(s.charAt(right), right);
                right++;
            }
            maxLength=Math.max(maxLength, right-left);
            //this is currently max length
            if(right==s.length()){
                break;
            }
            //找出与当前存储的重复的字符所在的位置索引index,并删除left~index的数据
            int repIndex=ch2PosMap.get(s.charAt(right));
            for(int i=left;i<=repIndex;i++){
                Character ch=pos2ChMap.remove(i);
                ch2PosMap.remove(ch);
            }
            //更新left索引
            left=repIndex+1;
        }
        return maxLength;
    }
~~~



**官方答案滑动窗口算法**

这道题主要用到思路是：滑动窗口

什么是滑动窗口？

其实就是一个队列,比如例题中的 $abcabcbb$，进入这个队列（窗口）为 $abc $满足题目要求，当再进入 $a$，队列变成了 $abca$，这时候不满足要求。所以，我们要移动这个队列！

如何移动？

我们只要把队列的左边的元素移出就行了，直到满足题目要求！

一直维持这样的队列，找出队列出现最长的长度时候，求出解！下面的解答并没有将重复的数据移除，而是通过更新索引的方式来更新滑动窗口的大小。

时间复杂度：$O(n)$

~~~java
public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for(int i = 0; i < s.length(); i ++){
            if(map.containsKey(s.charAt(i))){
                left = Math.max(left,map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-left+1);
        }
        return max;
        
    }
~~~



#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

难度中等3029收藏分享切换为英文接收动态反馈

给你一个包含 $n$ 个整数的数组 $nums$，判断 $nums$ 中是否存在三个元素 $a，b，c ，$使得 $a + b + c = 0$ ？请你找出所有和为 $0$ 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

 

**示例 1：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

```

**示例 2：**

```
输入：nums = []
输出：[]

```

**示例 3：**

```
输入：nums = [0]
输出：[]

```

 

**提示：**

- `0 <= nums.length <= 3000`
- `-105 <= nums[i] <= 105`

通过次数432,544提交次数1,389,942

**传统方法**

首先计算两个数的和，然后进行去重，最后合并三个数的和。（注意统计两个数的和的去重中previsouLeftVal的设置方式）。这种去重方式比较垃圾，后面

有相应的改进方法。

~~~java
 List<List<Integer>> twoSum(int[] nums,int target){
        List<List<Integer>> result=new ArrayList();
        if(nums.length<2){
            return result;
        }
        int left=0,right=nums.length-1;
        //to avoid repeat 
        Integer previousLeftVal=null;
        while(left<right){
            while(left<right&&Objects.equals(nums[left], previousLeftVal)){
                left++;
            }
            if(left==right){
                break;
            }
            int sum=nums[left]+nums[right];
            if(sum>target){
                right--;
            }
            if(sum<target){
                left++;
            }
            if(sum==target){
                List<Integer> tIntegers=new ArrayList(2);
                tIntegers.add(nums[left]);
                tIntegers.add(nums[right]);
                result.add(tIntegers);
                previousLeftVal=nums[left];
                left++;
                right--;
            }
        }
        return result;
    }
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result=new ArrayList();
        if(nums.length<3){
            return result;
        }
        Arrays.sort(nums);
        //key second element ,val third elements sets
        Map<Integer,Set<Integer>> secondThirdMap=new HashMap();
        for(int i=0;i+2<nums.length;i++){
            int[] rightNums=Arrays.copyOfRange(nums, i+1, nums.length);
            List<List<Integer>> twoSumResult=twoSum(rightNums,0-nums[i]);
            for(List<Integer> two:twoSumResult){
                if(secondThirdMap.containsKey(two.get(0))){
                    Set<Integer> thirdElement=secondThirdMap.get(two.get(0));
                    if(thirdElement.contains(two.get(1))){
                        continue;
                    }
                    thirdElement.add(two.get(1));
                }else{
                    Set<Integer> thirdElement=new HashSet();
                    thirdElement.add(two.get(1));
                    secondThirdMap.put(two.get(0), thirdElement);
                }
                List<Integer> thrIntegers=new ArrayList();
                thrIntegers.add(nums[i]);
                thrIntegers.addAll(two);
                result.add(thrIntegers);
            }
        }
        return result;
    }
~~~



**优化后代码**

~~~java
//no repeat two sum
    List<List<Integer>> twoSum(int[] nums,int target,int left,int right){
        List<List<Integer>> result=new ArrayList();
        if(nums.length<2){
            return result;
        }
        while(left<right){
            int sum=nums[left]+nums[right];
            if(sum>target){
                right--;
            }
            if(sum<target){
                left++;
            }
            if(sum==target){
                List<Integer> tIntegers=new LinkedList();
                tIntegers.add(nums[left]);
                tIntegers.add(nums[right]);
                result.add(tIntegers);
                //avoid repeat
                while(left<right&&nums[left+1]==nums[left]) left++;
                while(left<right&&nums[right-1]==nums[right]) right--;
                left++;
                right--;
            }
        }
        return result;
    }
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result=new ArrayList();
        if(nums.length<3){
            return result;
        }
        Arrays.sort(nums);
        for(int i=0;i+2<nums.length;i++){
            //no need to traverse
            if(nums[i]>0){
                break;
            }
            //to avoid repeat,assure first element is not same
            while(i>0&&i+2<nums.length&&nums[i]==nums[i-1]) i++;
            List<List<Integer>> twoSumResult=twoSum(nums,0-nums[i],i+1,nums.length-1);
            for(List<Integer> twoSumElements:twoSumResult){
                twoSumElements.add(0,nums[i]);
                result.add(twoSumElements);
            }
            
        }
        return result;
    }
~~~



**网上精简答案**

~~~java
 public List<List<Integer>> threeSum(int[] nums) {// 总时间复杂度：O(n^2)
        List<List<Integer>> ans = new ArrayList<>();
        if (nums == null || nums.length <= 2) return ans;

        Arrays.sort(nums); // O(nlogn)

        for (int i = 0; i < nums.length - 2; i++) { // O(n^2)
            if (nums[i] > 0) break; // 第一个数大于 0，后面的数都比它大，肯定不成立了
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 去掉重复情况
            int target = -nums[i];
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                if (nums[left] + nums[right] == target) {
                    ans.add(new ArrayList<>(Arrays.asList(nums[i], nums[left], nums[right])));
                    
                    // 现在要增加 left，减小 right，但是不能重复，比如: [-2, -1, -1, -1, 3, 3, 3], i = 0, left = 1, right = 6, [-2, -1, 3] 的答案加入后，需要排除重复的 -1 和 3
                    left++; right--; // 首先无论如何先要进行加减操作
                    while (left < right && nums[left] == nums[left - 1]) left++;
                    while (left < right && nums[right] == nums[right + 1]) right--;
                } else if (nums[left] + nums[right] < target) {
                    left++;
                } else {  // nums[left] + nums[right] > target
                    right--;
                }
            }
        }
        return ans;
    }
~~~



#### [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

难度中等1209收藏分享切换为英文接收动态反馈

运用你所掌握的数据结构，设计和实现一个  [LRU (最近最少使用) 缓存机制](https://baike.baidu.com/item/LRU) 。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以正整数作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。
- `void put(int key, int value)` 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。



**进阶**：你是否可以在 `O(1)` 时间复杂度内完成这两种操作？

 

**示例：**

```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4

```

 

**提示：**

- `1 <= capacity <= 3000`
- `0 <= key <= 3000`
- `0 <= value <= 104`
- 最多调用 `3 * 104` 次 `get` 和 `put`

通过次数137,331提交次数264,813

**我的解法**

采用LinkedList作为队列的基本结构，为了在O（1）的时间复杂度之内做put、get操作，使用map来记录每个元素的位置。

~~~java
class LRUCache {
    Deque<Pair> queue=new LinkedList();
    //to record pair position
    Map<Integer,Pair> map=new HashMap();
    int capacity;
    //current pair size;
    int size;
    //custom pair
    static class Pair{
        int key;
        int value;
        public Pair(int key,int value){
            this.key=key;
            this.value=value;
        }
    }
    public LRUCache(int capacity) {
        this.capacity=capacity;
    }
    
    public int get(int key) {
        Pair target=map.get(key);
        if(target==null){
            return -1;
        }
        setToHead(target,null);
        return target.value;
    }
    //set the target to the head of queueu
    private void setToHead(Pair toUpdatePair,Integer value){
        queue.remove(toUpdatePair);
        if(value!=null){
            toUpdatePair.value=value;
        }
        queue.addFirst(toUpdatePair);
    }
    
    public void put(int key, int value) {
        //update lru cache
        if(map.containsKey(key)){
            setToHead(map.get(key),value);
            return;
        }
        //lru remove the last pair from queue tail
        if(size==capacity){
            Pair last=queue.removeLast();
            map.remove(last.key);
            size--;
        }
        //insert to the head of the queue
        Pair pair=new Pair(key,value);
        queue.addFirst(pair);
        map.put(key,pair);
        size++;
    }
}
~~~



**官方版本**

采用虚拟节点的方式作为队头和队尾，能够减少很多复杂的逻辑判断。

~~~java
public class LRUCache {
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // 添加进哈希表
            cache.put(key, newNode);
            // 添加至双向链表的头部
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode tail = removeTail();
                // 删除哈希表中对应的项
                cache.remove(tail.key);
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}
~~~



#### [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

难度中等1007收藏分享切换为英文接收动态反馈

请你来实现一个 `myAtoi(string s)` 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 `atoi` 函数）。

函数 `myAtoi(string s)` 的算法如下：

- 读入字符串并丢弃无用的前导空格
- 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
- 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
- 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 `0` 。必要时更改符号（从步骤 2 开始）。
- 如果整数数超过 32 位有符号整数范围 $[−2^{31},  2^{31} − 1]$ ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 $−2^{31}$ 的整数应该被固定为 $−2^{31}$ ，大于 $2^{31} − 1$` 的整数应该被固定为 $2^{31} − 1$ 。
- 返回整数作为最终结果。

**注意：**

- 本题中的空白字符只包括空格字符 `' '` 。
- 除前导空格或数字后的其余字符串外，**请勿忽略** 任何其他字符。



**示例 1：**

```
输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
第 1 步："42"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："42"（读入 "42"）
           ^
解析得到整数 42 。
由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
```

**示例 2：**

```
输入：s = "   -42"
输出：-42
解释：
第 1 步："   -42"（读入前导空格，但忽视掉）
            ^
第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
             ^
第 3 步："   -42"（读入 "42"）
               ^
解析得到整数 -42 。
由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。

```

**示例 3：**

```
输入：s = "4193 with words"
输出：4193
解释：
第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
             ^
解析得到整数 4193 。
由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。

```

**示例 4：**

```
输入：s = "words and 987"
输出：0
解释：
第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
         ^
解析得到整数 0 ，因为没有读入任何数字。
由于 0 在范围 [-231, 231 - 1] 内，最终结果为 0 。
```

**示例 5：**

```
输入：s = "-91283472332"
输出：-2147483648
解释：
第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
          ^
第 3 步："-91283472332"（读入 "91283472332"）
                     ^
解析得到整数 -91283472332 。
由于 -91283472332 小于范围 [-231, 231 - 1] 的下界，最终结果被截断为 -231 = -2147483648 。
```

 

**提示：**

- `0 <= s.length <= 200`
- `s` 由英文字母（大写和小写）、数字（`0-9`）、`' '`、`'+'`、`'-'` 和 `'.'` 组成

通过次数256,304提交次数1,202,092

**我的答案**

题目本身不困难，困难的在于逻辑思维的严谨性，以及包括对于越界的处理。为了简化代码逻辑越界的处理，采用long类型，每次输入一个字符之后，判断数字是否越界。对于字符的处理，特殊字符包括输入符号$+,-$以及输入的空格、数字符号。使用两个变量$readDigit$以及$readSymbol$分别表示当前已经读取到了数字或者符号，那么后续的字符如果不是数字，那么直接跳出循环。代码如下：

~~~java
 public int myAtoi(String s) {
        long result = 0;
        boolean readSymbol = false, readDigit = false;
        int i = 0;
        if (s == null) {
            return (int) result;
        }
        boolean negative = false;
        boolean overflow = false;
        for (; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (readSymbol || readDigit) {
                if (!Character.isDigit(ch)) {
                    break;
                } else {
                    result = result * 10 + ch - '0';
                }
            } else {
                if (ch == '-') {
                    readSymbol = true;
                    negative = true;
                } else if (ch == '+') {
                    readSymbol = true;
                } else if (Character.isDigit(ch)) {
                    //first read digit
                    readDigit = true;
                    result = ch - '0';
                }else if(ch==' '){
                    //read space,continue
                    continue;
                } else {
                    break;
                }
            }
            if (negative && 0 - result < Integer.MIN_VALUE) {
                overflow = true;
                break;
            } else if (!negative && result > Integer.MAX_VALUE) {
                overflow = true;
                break;
            }
        }
        if (overflow) {
            return negative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        }
        return negative ? (int)(0 - result) :(int) result;
    }
~~~



**网上答案**

注意溢出条件的判断

~~~java
public int myAtoi(String s) {
       if (s.length()==0) {
			return 0;
		}
		int res=0;
		int status=0;		
		int flag=1;
		for (int i = 0; i < s.length(); i++) {
			char a=s.charAt(i);
			if (a==' '&&status==0) {
				continue;
			}else if((a=='-'||a=='+')&&status==0){
				if (a=='-') {
					flag=-1;					
				}
				status=1;//后面必须是数字
			}else if(a<='9'&&a>='0'){
				status=1;
				if (res>(Integer.MAX_VALUE-a+'0')/10) {
					res=  Integer.MAX_VALUE;
				}else 
				if (res<(Integer.MIN_VALUE+a-'0')/10) {
					res=  Integer.MIN_VALUE;
				}else{
					res=res*10+(a-'0')*flag;		
				}		
			}else{
				break;
			}
		}
		
        return res;
    
    }
~~~



#### [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

难度中等508收藏分享切换为英文接收动态反馈

罗马数字包含以下七种字符： `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做  `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

 

**示例 1:**

```
输入: 3
输出: "III"
```

**示例 2:**

```
输入: 4
输出: "IV"
```

**示例 3:**

```
输入: 9
输出: "IX"
```

**示例 4:**

```
输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.

```

**示例 5:**

```
输入: 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

 

**提示：**

- `1 <= num <= 3999`

通过次数145,786提交次数224,614

**我的解答**

关键是考虑问题比较全面，思路比较缜密，优先采用递归方法。记得递归的每一层判断都要return。

~~~java
ublic void intToRoman(int num,StringBuilder result){
        if(num==0){
            return;
        }
        int i=0;
        if(num>=1000){
            for(i=0;i<num/1000;i++){
                result.append("M");
            }
            intToRoman(num-i*1000,result);
            return;
        }
        if(num>=900){
            result.append("CM");
            intToRoman(num-900,result);
            return;
        }
        if(num>=500){
            result.append("D");
            intToRoman(num-500,result);
            return;
        }
        if(num>=400){
            result.append("CD");
            intToRoman(num-400,result);
            return;
        }
        if(num>=100){
            for(i=0;i<num/100;i++){
                result.append("C");
            }
            intToRoman(num-i*100,result);
            return;
        }
        if(num>=90){
            result.append("XC");
            intToRoman(num-90,result);
            return;
        }
        if(num>=50){
            result.append("L");
            intToRoman(num-50,result);
            return;
        }
        if(num>=40){
            result.append("XL");
            intToRoman(num-40,result);
            return;
        }
        if(num>=10){
            for(i=0;i<num/10;i++){
                result.append("X");
            }
            intToRoman(num-i*10,result);
            return;
        }
        if(num>=9){
            result.append("IX");
            return;
        }
        if(num>=5){
            result.append("V");
            intToRoman(num-5,result);
            return;
        }
        if(num>=4){
            result.append("IV");
            return;
        }
        if(num>=1){
            for(i=0;i<num;i++){
                result.append("I");
            }
            return;
        }

    }
    public String intToRoman(int num) {
        StringBuilder result=new StringBuilder();
        intToRoman(num,result);
        return result.toString();
    }
~~~



**精简解答**

~~~java
public String intToRoman(int s){
        //题目里只给了七个基本的，但是间隔里的数也可补充上。
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};    
        String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        int i = 0;
        StringBuilder result = new StringBuilder();
        while(s > 0 && i < values.length){
            //这里用while，因为可能存在III，重复使用
            while(s >= values[i]){
                s = s - values[i];
                result.append(symbols[i]);
            }
            i++;
        }
        return result.toString();

    }
~~~



#### [17优. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

难度中等1164收藏分享切换为英文接收动态反馈

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png)

 

**示例 1：**

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

```

**示例 2：**

```
输入：digits = ""
输出：[]

```

**示例 3：**

```
输入：digits = "2"
输出：["a","b","c"]

```

 

**提示：**

- `0 <= digits.length <= 4`
- `digits[i]` 是范围 `['2', '9']` 的一个数字。

通过次数228,525提交次数407,623

**耗时较高方法**

主要耗时是string拼接以及map映射

~~~java
static Map<Character,String> digit2AlphaMap=new HashMap();
    static{
        digit2AlphaMap.put('2',"abc");
        digit2AlphaMap.put('3', "def");
        digit2AlphaMap.put('4', "ghi");
        digit2AlphaMap.put('5', "jkl");
        digit2AlphaMap.put('6', "mno");
        digit2AlphaMap.put('7', "pqrs");
        digit2AlphaMap.put('8', "tuv");
        digit2AlphaMap.put('9', "wxyz");
    }
    public List<String> letterStrings(String digits,List<String> preResult){
        if(digits.isEmpty()){
            return preResult;
        }
        List<String> result=new ArrayList();
        String alpha=digit2AlphaMap.get(digits.charAt(0));
        for(int i=0;i<alpha.length();i++){
            if(preResult.isEmpty()){
                result.add(String.valueOf(alpha.charAt(i)));
            }else{
                for(String com:preResult){
                    result.add(com+alpha.charAt(i));
                }
            }
        }
        return letterStrings(digits.substring(1),result);
    }
    public List<String> letterCombinations(String digits) {
        if(digits==null||digits.isEmpty()){
            return new ArrayList();
        }
        List<String> result=letterStrings(digits, new ArrayList());
        return result;
    }
~~~



**优化方法**

不采用map结构做映射，而是采用数组结构，不采用string拼接而是采用StringBuilder，结合回溯法和dfs实现。

~~~java
 public List<String> letterCombinations(String digits) {
        
        List<String> ans = new ArrayList<String>();
        int len = digits.length();
        if(digits==null || digits.length()==0) return ans;
        iterStr(digits, ans, new StringBuilder(),0);
        return ans; 
    }
    public String[] map = {"abc", "def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    public void iterStr(String digits, List<String> ans, StringBuilder buff, int index){
        if(index==digits.length()){
            ans.add(buff.toString());
            return;
        }
        int in = digits.charAt(index)-'0'-2;
        String s= map[in];
        int len=s.length();
        for(int i =0; i<len;i++){
            buff.append(s.charAt(i));
            iterStr(digits, ans, buff,index+1);
            buff.deleteCharAt(buff.length()-1);
        }
    }
~~~



#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

难度中等1254收藏分享切换为英文接收动态反馈

给你一个链表，删除链表的倒数第 `n`* *个结点，并且返回链表的头结点。

**进阶：**你能尝试使用一趟扫描实现吗？

 

**示例 1：**

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

```

**示例 2：**

```
输入：head = [1], n = 1
输出：[]

```

**示例 3：**

```
输入：head = [1,2], n = 1
输出：[1]

```

 

**提示：**

- 链表中结点的数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

通过次数332,595提交次数807,739

**解答**

题目要求只需要一趟扫描，对于长度为$n$的单链表，删除倒数第$k(k<=n)$来说，倒数$k$个就相当于正数第$n-k+1$(索引从1开始)个节点，由于是单链表，要删除一个节点需要记录该节点的前驱节点，所有相当于找到第$n-k$个节点。使用两个指针，第一个指针$p$先走$k$次，即达到第$k$个节点，然后指针$p,q$一起走，这一次走了$n-k$次，循环跳出之后$q$就指向第$n-k$个节点，考虑到$k==n$的情形需要做特殊判断。

~~~java
public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head==null){
            return head;
        }
        ListNode p=head,q=null;
        //for the kth last node equals the n-k+1 node
        int pSeq=1;
        //current p point the kth node
        while(p!=null&&pSeq<=n){
            pSeq++;
            p=p.next;
        }
        //after break ,q point the n-k th node,if k equals n,q point null
        while(p!=null){
            p=p.next;
            if(q==null){
                q=head;
            }else{
                q=q.next;
            }
        }
        //delete node is the head
        if(q==null){
            return head.next;
        }else{
            //remove q.next node,that's to say the n-k+1 th node
            q.next=q.next.next;
        }
        return head;
    }
~~~



**官方答案**

采用虚拟节点，减少判断

~~~java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummyHead = new ListNode(0);
    dummyHead.next = head;

    // 慢指针初始指向虚拟头结点
    ListNode slow = dummyHead;
    // 快指针初始指向虚拟头结点
    ListNode fast = dummyHead;

    // 快指针先向前移动n+1步
    for(int i = 0; i <= n; i++) {
        fast = fast.next;
    }

    // 快慢指针同时向前移动，直到快指针指向null
    while (fast!=null){
        fast = fast.next;
        slow = slow.next;
    }

    // 慢指针的下一个节点即待删除节点
    ListNode delNode = slow.next;
    // 慢指针的后继指针指向待删除节点的下一个节点
    // 这样就将待删除节点删除了
    slow.next = delNode.next;
    delNode.next = null;
    return dummyHead.next;
}
~~~



#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

难度中等1596收藏分享切换为英文接收动态反馈

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的 **括号组合。

 

**示例 1：**

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

```

**示例 2：**

```
输入：n = 1
输出：["()"]

```

 

**提示：**

- `1 <= n <= 8`

通过次数235,756提交次数306,515

**回溯方法**

回溯方法的经典实用，每次迭代其实要么是添加左括号要么添加右括号，想要生成正确的符号匹配需要满足特定的规则。首先是当前匹配的左括号数量不能低于右括号数量，第二最多左括号个数是$n$。注意每次递归之后要删除添加的左括号或右括号，这叫做回溯。为了提高效率采用$StringBuilder$做字符串拼接。

~~~
List<String> result = new LinkedList();
    public void generate(StringBuilder ans, int totalLeftBracket, int unMatchLeftBracket, int n) {
        if (ans.length() == 2 * n) {
            result.add(ans.toString());
            return;
        }
        if (totalLeftBracket == n) {
            //can only match right bracket )
            generate(ans.append(")"), totalLeftBracket, unMatchLeftBracket - 1, n);
            ans.deleteCharAt(ans.length()-1);
        } else {
            //can only match left bracket (
            if (unMatchLeftBracket == 0) {
                generate(ans.append("("), totalLeftBracket + 1, unMatchLeftBracket + 1, n);
                ans.deleteCharAt(ans.length()-1);
            } else {
                //match left bracket;
                generate(ans.append("("), totalLeftBracket + 1, unMatchLeftBracket + 1, n);
                //backtrace
                ans.deleteCharAt(ans.length() - 1);
                //match right bracket
                generate(ans.append(")"), totalLeftBracket , unMatchLeftBracket - 1, n);
                ans.deleteCharAt(ans.length()-1);
            }
        }
    }

    public List<String> generateParenthesis(int n) {
        generate(new StringBuilder(), 0, 0, n);
        return result;
    }
~~~



**好的解答**

~~~java
 public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<String>();
        generate(res, "", 0, 0, n);
        
        return res;
    }
        //count1统计“(”的个数，count2统计“)”的个数
    public void generate(List<String> res , String ans, int count1, int count2, int n){
        
        if(count1 > n || count2 > n) return;
        
        if(count1 == n && count2 == n)  res.add(ans);
 
        
        if(count1 >= count2){
            String ans1 = new String(ans);
            generate(res, ans+"(", count1+1, count2, n);
            generate(res, ans1+")", count1, count2+1, n);
            
        }
~~~



#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

难度中等1206收藏分享切换为英文接收动态反馈

整数数组 $nums$ 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，$nums$ 在预先未知的某个下标 $k（0 <= k < nums.length）$上进行了 **旋转**，使数组变为 $[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标从 0 开始 计数）$。例如， $[0,1,2,4,5,6,7]$ 在下标 $3$ 处经旋转后可能变为 $[4,5,6,7,0,1,2]$ 。

给你 **旋转后** 的数组 $nums$ 和一个整数 $target$ ，如果 $nums$ 中存在这个目标值 $target$ ，则返回它的索引，否则返回 $-1$ 。

 

**示例 1：**

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**示例 3：**

```
输入：nums = [1], target = 0
输出：-1

```

 

**提示：**

- $1 <= nums.length <= 5000$
- $-10^4 <= nums[i] <= 10^4$
- $nums$中的每个值都 **独一无二**
- $nums$ 肯定会在某个点上旋转
- $-10^4 <= target <= 10^4$



**进阶：**你可以设计一个时间复杂度为 $O(log n)$ 的解决方案吗？

通过次数228,110提交次数560,441

**二分查找变形**

该题其实就是二分查找的变形，首先要将该搜索转移到二分查找，二分查找要求序列范围内的数据是排序的，如何判断呢。通过前后的两个元素的值来判断，记得判断符号包括$==$号，如果当前范围左边的元素不大于最右边的，那么认为是有序的，因为如果是反转元素如果在这个区间范围内的话，左边的一定大于右边的。

~~~java
 //binary search the target, if exists ret it's index else return -1
    public int binarySearch(int[] nums,int left,int right,int target){
        int low=left,high=right;
        int mid=0;
        while(low<=high){
            mid=(low+high)/2;
            if(target==nums[mid]){
                return mid;
            }
            if(target>nums[mid]){
                low=mid+1;
            }else{
                high=mid-1;
            }
        }
        return -1;
    }
    public int search(int[] nums,int target,int left,int right){
        if(left>right){
            return -1;
        }
        boolean sorted=nums[left]<=nums[right];
        if(sorted){
            return binarySearch(nums, left, right, target);
        }
        //not sorted,split the nums array
        int mid=(left+right)/2;
        int leftSearchIndex=search(nums, target,left,mid);
        if(leftSearchIndex!=-1){
            return leftSearchIndex;
        }
        return search(nums,target,mid+1,right);
    }
    public int search(int[] nums, int target) {
        return search(nums,target,0,nums.length-1);
    }
~~~



#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

难度中等516收藏分享切换为英文接收动态反馈

给定一个数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的每个数字在每个组合中只能使用一次。

**说明：**

- 所有数字（包括目标数）都是正整数。
- 解集不能包含重复的组合。 

**示例 1:**

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

```

**示例 2:**

```
输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
```

通过次数135,569提交次数211,713

**解答**

注意去重条件以及回溯问题

~~~java
  List<List<Integer>> result=new LinkedList();
    public void combinationSum(int[] candidates,int startIndex,int target,List<Integer> ans){
        if(target==0){
            result.add(new ArrayList(ans));
            //do not ret ,may be from startIndex exists 0 val
        }
        for(int i=startIndex;i<candidates.length;i++){
            //ascending order,no target
            if(target<candidates[i]){
                break;
            }
            //to avoid repeat
            if(i!=startIndex&&candidates[i-1]==candidates[i]){
                continue;
            }
            int length=ans.size();
            ans.add(candidates[i]);
            combinationSum(candidates, i+1, target-candidates[i], ans);
            //remove last element
            ans.remove(length);
        }
    }
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        combinationSum(candidates, 0, target,new ArrayList());
        return result;
    }
~~~



#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

难度中等1181收藏分享切换为英文接收动态反馈

给定一个** 没有重复** 数字的序列，返回其所有可能的全排列。

**示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

通过次数265,425提交次数341,749

**传统笨方法**

使用set存储还未访问过的数据，这种方式一个缺点就是会有大量的数据复制操作

~~~java
ist<List<Integer>> result = new LinkedList();
	//set表示当前未访问的数据
    public void permute(Set<Integer> set, LinkedList<Integer> ans) {
        if (set.isEmpty()) {
            if (!ans.isEmpty()) {
                result.add(new ArrayList(ans));
            }
            return;
        }
        Iterator<Integer> iterator = set.iterator();
        while (iterator.hasNext()) {
            Integer val = iterator.next();
            Set<Integer> copySet = new HashSet(set);
            copySet.remove(val);
            ans.add(val);
            permute(copySet, ans);
            ans.removeLast();
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        Set<Integer> set = new HashSet();
        for (int num : nums) {
            set.add(num);
        }
        permute(set, new LinkedList());
        return result;
    }
~~~



**优化方法2避免数组复制**

使用空间换时间的方法，优化方法可以考虑使用标志位数组判断是否已访问

~~~java
	List<List<Integer>> result = new LinkedList();
    //ans和deque存储相同的数据，ans是用来做数据去重判断的
    public void permute(int[] nums,Deque<Integer> deque, Set<Integer> ans){
        if(deque.size()==nums.length){
            result.add(new ArrayList(deque));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(ans.contains(nums[i])){
               continue; 
            }
            ans.add(nums[i]);
            deque.add(nums[i]);
            permute(nums,deque,ans);
            ans.remove(nums[i]);
            deque.removeLast();
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        permute(nums,new LinkedList(),new HashSet());
        return result;
    }
~~~

**官方版本，通过交换索引避免记录访问过的数**

~~~java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();

        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }
}

~~~



#### [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

难度中等672收藏分享切换为英文接收动态反馈

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

**示例:**

```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```

**说明：**

- 所有输入均为小写字母。
- 不考虑答案输出的顺序。

通过次数168,516提交次数257,090

**传统按照map进行分组的方法**

效率较低

~~~java
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result=new ArrayList();
        Map<String,List<String>>  map=new HashMap();
        for(String str:strs){
            char[] chArr=str.toCharArray();
            Arrays.sort(chArr);
            String sortStr=new String(chArr);
            List<String> groupList=map.get(sortStr);
            if(groupList==null){
                groupList=new LinkedList();
                map.put(sortStr,groupList);
                result.add(groupList);
            }
            groupList.add(str);
        }
        return result;
    }
~~~



**官方版本（巧妙利用hash映射关系）**

~~~java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            int[] counts = new int[26];
            int length = str.length();
            for (int i = 0; i < length; i++) {
                counts[str.charAt(i) - 'a']++;
            }
            // 将每个出现次数大于 0 的字母和出现次数按顺序拼接成字符串，作为哈希表的键
            StringBuffer sb = new StringBuffer();
            for (int i = 0; i < 26; i++) {
                if (counts[i] != 0) {
                    sb.append((char) ('a' + i));
                    sb.append(counts[i]);
                }
            }
            String key = sb.toString();
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
~~~



#### [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

难度中等600收藏分享切换为英文接收动态反馈

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。

 

**示例 1：**

```
输入：x = 2.00000, n = 10
输出：1024.00000

```

**示例 2：**

```
输入：x = 2.10000, n = 3
输出：9.26100

```

**示例 3：**

```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25

```

 

**提示：**

- `-100.0 < x < 100.0`
- `-231 <= n <= 231-1`
- `-104 <= xn <= 104`

通过次数160,839提交次数430,917

**比较简单，不详细说明了**

~~~java
public double myPow(double x, int n) {
        if(n==0){
            return 1;
        }
        if(n==-1){
            return 1/x;
        }
        if(n==1){
            return x;
        }
        double halfVal=myPow(x, n/2);
        return halfVal*halfVal*myPow(x, n-n/2*2);
    }
~~~



#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

难度中等825收藏分享切换为英文接收动态反馈

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

 

**示例 1：**

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

```

**示例 2：**

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

 

**提示：**

- `1 <= intervals.length <= 104`
- `intervals[i].length == 2`
- `0 <= starti <= endi <= 104`

通过次数197,166提交次数442,513

**我的解法**

先进行一维数组的排序，然后再进行区间合并，一维数组按照首个元素进行排序，注意最后一个元素的处理。

~~~java
 public int[][] merge(int[][] intervals) {
        //sort by first index of interval
        Arrays.sort(intervals,(o1,o2)->o1[0]==o2[0]?o1[1]-o2[1]:o1[0]-o2[0]);
        int left=intervals[0][0],right=intervals[0][1];
        List<int[]> intervalList=new ArrayList();
        for(int i=1;i<intervals.length;i++){
            int[] interval=intervals[i];
            //create new interval,merge previous interval
            if(interval[0]>right){
                intervalList.add(new int[]{left,right});
                left=interval[0];
                right=interval[1];
            }else{
                //merge interval
                right=Math.max(right, interval[1]);
            }
        }
        //merge final interval
        intervalList.add(new int[]{left,right});
        int[][] mergeIntervals=new int[intervalList.size()][2];
        int i=0;
        for(int[] interval:intervalList){
            mergeIntervals[i++]=interval;
        }
        return mergeIntervals;
    }
~~~



**官方版本**

~~~java
public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
~~~



#### [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)

难度中等377收藏分享切换为英文接收动态反馈

给你一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

 

**示例 1：**

```
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]

```

**示例 2：**

```
输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

**示例 3：**

```
输入：intervals = [], newInterval = [5,7]
输出：[[5,7]]

```

**示例 4：**

```
输入：intervals = [[1,5]], newInterval = [2,3]
输出：[[1,5]]

```

**示例 5：**

```
输入：intervals = [[1,5]], newInterval = [2,7]
输出：[[1,7]]

```

 

**提示：**

- `0 <= intervals.length <= 104`
- `intervals[i].length == 2`
- `0 <= intervals[i][0] <= intervals[i][1] <= 105`
- `intervals` 根据 `intervals[i][0]` 按 **升序** 排列
- `newInterval.length == 2`
- `0 <= newInterval[0] <= newInterval[1] <= 105`

通过次数64,668提交次数160,273

**我的解答**

相对来说，比较简单，跟上一题的区间合并比较相似，新插入的区间保证要在合并之后的区间的合适位置（即保证合并后的区间满足按照左边第一个元素非降序排列）

~~~java
public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> intervalList=new ArrayList();
        if(intervals.length==0){
            intervalList.add(newInterval);
        }else{
            boolean insertSuccess=false;
            //insert newInterval to suitable position
            for(int i=0;i<intervals.length;i++){
                int[] interval=intervals[i];
                if(!insertSuccess&&interval[0]>=newInterval[0]){
                    intervalList.add(newInterval);
                    insertSuccess=true;
                }
                intervalList.add(interval);
            }
            if(!insertSuccess){
                intervalList.add(newInterval);
            }
        }
        //merge interval
        List<int[]> mergeIntervalList=new ArrayList();
        int left=intervalList.get(0)[0],right=intervalList.get(0)[1];
        for(int i=1;i<intervalList.size();i++){
            int[] interval=intervalList.get(i);
            if(interval[0]>right){
                mergeIntervalList.add(new int[]{left,right});
                left=interval[0];
                right=interval[1];
            }else{
                right=Math.max(right,interval[1]);
            }
        }
        mergeIntervalList.add(new int[]{left,right});
        return mergeIntervalList.toArray(new int[mergeIntervalList.size()][]);
    }
~~~



**优化方法（只遍历一次）**

不再采用插入的方法，而是采用边遍历边更新区间的方法。

~~~java
public int[][] insert(int[][] intervals, int[] newInterval) {
         //merge interval
        List<int[]> mergeIntervalList=new ArrayList();
        if(intervals.length==0){
            mergeIntervalList.add(newInterval);
        }else{
            boolean insertSuccess=false;
            int left,right;
            //init
            if(intervals[0][0]<=newInterval[0]){
                left=intervals[0][0];
                right=intervals[0][1];
            }else{
                left=newInterval[0];
                right=newInterval[1];
            }
            for(int i=0;i<intervals.length;i++){
                int[] interval=intervals[i];
                if(!insertSuccess&&interval[0]>=newInterval[0]){
                    interval=newInterval;
                    insertSuccess=true;
                    i--;
                }
                if(interval[0]>right){
                    mergeIntervalList.add(new int[]{left,right});
                    left=interval[0];
                    right=interval[1];
                }else{
                    right=Math.max(right,interval[1]);
                }
            }
            //newIntervals left index is biggest
            if(!insertSuccess){
                if(right<newInterval[0]){
                    mergeIntervalList.add(new int[]{left,right});
                    left=newInterval[0];
                    right=newInterval[1];
                }else{
                    right=Math.max(right,newInterval[1]);
                }
            }
            mergeIntervalList.add(new int[]{left,right});
        }
        return mergeIntervalList.toArray(new int[mergeIntervalList.size()][]);
    }
~~~



#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

难度中等507收藏分享切换为英文接收动态反馈

一个机器人位于一个 *m x n *网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png)

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

 

**示例 1：**

```
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

```

**示例 2：**

```
输入：obstacleGrid = [[0,1],[0,0]]
输出：1

```

 

**提示：**

- `m == obstacleGrid.length`
- `n == obstacleGrid[i].length`
- `1 <= m, n <= 100`
- `obstacleGrid[i][j]` 为 `0` 或 `1`

通过次数126,070提交次数334,074

**解法**

常规的动态规划方法，转移方程如下所示


$$
opt[m][n]=grid[m][n]==0?(opt[m][n-1]+opt[m-1][n],0)
$$

其中函数$opt[m][n]$表示从$grid[0][0]$出发到达$grid[m][n]$的路径次数，注意只有当位置$grid[m][n]!=1$即不是障碍物的情况下，才可达。注意数组的初始化

~~~java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m=obstacleGrid.length;
        int n=obstacleGrid[0].length;
        int[][] ans=new int[m][n];
        int i,j;
        for(i=0;i<m;i++){
            if(i==0){
                ans[i][0]=obstacleGrid[i][0]==0?1:0;
            }else{
                ans[i][0]=obstacleGrid[i][0]==0?ans[i-1][0]:0;
            }
        }
        for(j=1;j<n;j++){
            ans[0][j]=obstacleGrid[0][j]==0?ans[0][j-1]:0;
        }
        for(i=1;i<m;i++){
            for(j=1;j<n;j++){
                int maxSum=ans[i-1][j]+ans[i][j-1];
                ans[i][j]=obstacleGrid[i][j]==0?maxSum:0;
            }
        }
        return ans[m-1][n-1];
    }
~~~



**减少空间的方法**

只记录前两行的数据，即使用数据$ans[2][n]$来进行数据统计

~~~java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m=obstacleGrid.length;
        int n=obstacleGrid[0].length;
        int[][] ans=new int[2][n];
        int i,j;
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                if(i==0){
                    if(j==0){
                        ans[0][j]=obstacleGrid[i][j]==0?1:0;
                        continue;
                    }
                    ans[0][j]=obstacleGrid[i][j]==0?ans[0][j-1]:0;
                    continue;
                }
                if(j==0){
                    ans[1][j]=obstacleGrid[i][j]==0?ans[0][j]:0;
                    continue;
                }
                int maxSum=ans[0][j]+ans[1][j-1];
                ans[1][j]=obstacleGrid[i][j]==0?maxSum:0;
            }
            if(i>=1){
                for(j=0;j<n;j++){
                    ans[0][j]=ans[1][j];
                }
            }
          
        }
        return m>=2?ans[1][n-1]:ans[0][n-1];
    }
~~~



**官方答案（滚动数组解法）**

「滚动数组思想」是一种常见的动态规划优化方法，在我们的题目中已经多次使用到，例如「剑指 Offer 46. 把数字翻译成字符串」、「70. 爬楼梯」等，当我们定义的状态在动态规划的转移方程中只和某几个状态相关的时候，就可以考虑这种优化方法，目的是给空间复杂度「降维」。如果你还不知道什么是「滚动数组思想」，一定要查阅相关资料进行学习哦。

代码中给出了使用「滚动数组思想」优化后的实现。

回顾这道题，其实这类动态规划的题目在题库中也出现过多次，例如「221. 最大正方形」、「1162. 地图分析」等。他们都以二维坐标作为状态，大多数都可以使用滚动数组进行优化。如果我们熟悉这类问题，可以一眼看出这是一个动态规划问题。当我们不熟悉的时候，怎么想到用动态规划来解决这个问题呢？我们需要从问题本身出发，寻找一些有用的信息，例如本题中：$(i, j)$ 位置只能从 $(i - 1, j) 和 (i, j - 1)$走到，这样的条件就是在告诉我们这里转移是 「无后效性」 的，$f(i, j)$ 和任何的 $f(i', j')(i' > i, j' > j)$无关。
 >动态规划的题目分为两大类，一种是求最优解类，典型问题是背包问题，另一种就是计数类，比如这里的统计方案数的问题，它们都存在一定的递推性质。前者的递推性质还有一个名字，叫做 「最优子结构」 ——即当前问题的最优解取决于子问题的最优解，后者类似，当前问题的方案数取决于子问题的方案数。所以在遇到求方案数的问题时，我们可以往动态规划的方向考虑。
 >通常如果我们察觉到了这两点要素，这个问题八成可以用动态规划来解决。读者可以多多练习，熟能生巧。



~~~java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.length, m = obstacleGrid[0].length;
        int[] f = new int[m];

        f[0] = obstacleGrid[0][0] == 0 ? 1 : 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (obstacleGrid[i][j] == 1) {
                    f[j] = 0;
                    continue;
                }
                if (j - 1 >= 0 && obstacleGrid[i][j - 1] == 0) {
                    f[j] += f[j - 1];
                }
            }
        }
        
        return f[m - 1];
    }
~~~



#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

难度中等939收藏分享切换为英文接收动态反馈

在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

**示例 1:**

```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5

```

**示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

**说明:**

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

通过次数277,271提交次数428,410

**快速排序实现**

随机快排方式

~~~java
public void swapNodes(int[] nums,int i,int j){
        int tmp=nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
    }
    //using quick sort to find the kth larget number
    public int quickSortSearch(int[] nums,int left,int right,int k){
        if(left>right){
            return -1;
        }
        if(left==right){
            if(left==k){
                return left;
            }
            return -1;
        }
        Random random=new Random();
        //heap sort
        int seedIndex=left+random.nextInt(right-left+1);
        int seedNum=nums[seedIndex];
        int i=left-1;
        for(int j=left;j<=right;j++){
            if(nums[j]<=seedNum){
                swapNodes(nums, j, ++i);
                if(nums[i]==seedNum){
                    seedIndex=i;
                }
            }
        }
        swapNodes(nums, seedIndex, i);
        if(i==k){
            return i;
        }else if(i<k){
            return quickSortSearch(nums, i+1, right, k);
        }else{
            return quickSortSearch(nums, left, i-1, k);
        }
    }
    public int findKthLargest(int[] nums, int k) {
        int index=quickSortSearch(nums, 0, nums.length-1,nums.length-k);
        return index==-1?0:nums[index];
    }
~~~



**官方解答（底层也是随机快排实现）**

时间渐进复杂度是$O(n)$

~~~java
class Solution {
    Random random = new Random();

    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int quickSelect(int[] a, int l, int r, int index) {
        int q = randomPartition(a, l, r);
        if (q == index) {
            return a[q];
        } else {
            return q < index ? quickSelect(a, q + 1, r, index) : quickSelect(a, l, q - 1, index);
        }
    }

    public int randomPartition(int[] a, int l, int r) {
        int i = random.nextInt(r - l + 1) + l;
        swap(a, i, r);
        return partition(a, l, r);
    }

    public int partition(int[] a, int l, int r) {
        int x = a[r], i = l - 1;
        for (int j = l; j < r; ++j) {
            if (a[j] <= x) {
                swap(a, ++i, j);
            }
        }
        swap(a, i + 1, r);
        return i + 1;
    }

    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}
~~~



#### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

难度中等815收藏分享切换为英文接收动态反馈

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

 

**示例 1：**

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12

```

 

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 200`
- `0 <= grid[i][j] <= 100`

通过次数189,042提交次数277,094

**简单的动态规划实现**

~~~java
public int minPathSum(int[][] grid) {
        int m=grid.length;
        int n=grid[0].length;
        int[][] opt=new int[m][n];
        int i=0,j=0;
        int sum=0;
        for(;i<m;i++){
            sum+=grid[i][0];
            opt[i][0]=sum;
        }
        sum=0;
        for(j=0;j<n;j++){
            sum+=grid[0][j];
            opt[0][j]=sum;
        }
        for(i=1;i<m;i++){
            for(j=1;j<n;j++){
                opt[i][j]=grid[i][j]+Math.min(opt[i][j-1],opt[i-1][j]);
            }
        }
        return opt[m-1][n-1];
    }
~~~



#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

难度中等366收藏分享切换为英文接收动态反馈

假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 `[0,1,2,4,5,6,7]` ****可能变为 `[4,5,6,7,0,1,2]` 。

请找出其中最小的元素。

 

**示例 1：**

```
输入：nums = [3,4,5,1,2]
输出：1

```

**示例 2：**

```
输入：nums = [4,5,6,7,0,1,2]
输出：0

```

**示例 3：**

```
输入：nums = [1]
输出：1

```

 

**提示：**

- `1 <= nums.length <= 5000`
- `-5000 <= nums[i] <= 5000`
- `nums` 中的所有整数都是 **唯一** 的
- `nums` 原来是一个升序排序的数组，但在预先未知的某个点上进行了旋转

通过次数109,773提交次数209,345

**解答**

~~~java
 public int findMin(int[] nums,int left,int right){
        if(left==right){
            return nums[left];
        }
        //sorted array
        if(nums[left]<nums[right]){
            return nums[left];
        }
        int mid=(left+right)/2;
        int leftMin=findMin(nums,left,mid);
        int rightMin=findMin(nums,mid+1,right);
        return Math.min(leftMin,rightMin);
    }
    public int findMin(int[] nums) {
        return findMin(nums,0,nums.length-1);
    }
~~~



**迭代方法**

~~~java
public static int findMin(int[] nums) {
        int l = 0, r = nums.length - 1, mid = 0;
        while (l < r) {
            mid = l + ((r - l) >> 1);
            //这里有个编程技巧 
            //因为l<r 所以最后一轮肯定是(r,r+1)
            //那么mid 肯定是取值l 当判断条件是mid与l比时 会出现与自身比 造成出现等于情况 不好判断
            //所以判断条件时mid 与 r比 这样肯定是不同的两个数比
            if (nums[mid] < nums[r]) {  // mid可能为最小值
                r = mid;
            } else { // 没有重复值
                l = mid+1;      // mid肯定不是最小值
            }
        }
        return nums[mid];
    }
~~~



#### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

难度中等1044收藏分享切换为英文接收动态反馈

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

**进阶：**

- 你可以在 `O(n log n)` 时间复杂度和常数级空间复杂度下，对链表进行排序吗？



**示例 1：**

```
输入：head = [4,2,1,3]
输出：[1,2,3,4]

```

**示例 2：**

```
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]

```

**示例 3：**

```
输入：head = []
输出：[]

```

 

**提示：**

- 链表中节点的数目在范围 `[0, 5 * 104]` 内
- `-105 <= Node.val <= 105`

通过次数146,339提交次数217,060

**我的解法**

传统的分治方法实现，在计算一个单链表的中间节点的时候，需要注意一下，如果该单链表的长度是偶数，那么返回的中间节点应该是左边界的值，比如对于单链表$1->3$来说，应该返回的是$1$这个节点，而不是$3$这个节点（如果返回$3$会导致无限递归）。返回中间节点代码

~~~java
 public ListNode getMiddleNode(ListNode head){
        if(head==null||head.next==null){
            return head;
        }
        ListNode p=head,q=head.next;
        while(q!=null&&q.next!=null){
            p=p.next;
            q=q.next.next;
        }
        return p;
    }
~~~



完整代码

~~~java
//get mid position of single list
    public ListNode getMiddleNode(ListNode head){
        if(head==null||head.next==null){
            return head;
        }
        ListNode p=head,q=head.next;
        while(q!=null&&q.next!=null){
            p=p.next;
            q=q.next.next;
        }
        return p;
    }
    public ListNode sortList(ListNode head) {
      if(head==null||head.next==null){
          return head;
      }
      ListNode midNode=getMiddleNode(head);
      //divide node into two parts
      ListNode rightFirstNode=midNode.next;
      midNode.next=null;
      ListNode leftSortNode=sortList(head);
      ListNode rightSortNode=sortList(rightFirstNode);
      //combine two sorted list node, with a virtual node as head node
      ListNode dummyNode=new ListNode();
      ListNode p=dummyNode;
      while(leftSortNode!=null||rightSortNode!=null){
          if(leftSortNode!=null&&rightSortNode!=null){
              if(leftSortNode.val<=rightSortNode.val){
                  p.next=leftSortNode;
                  p=leftSortNode;
                  leftSortNode=leftSortNode.next;
              }else{
                  p.next=rightSortNode;
                  p=rightSortNode;
                  rightSortNode=rightSortNode.next;
              }
              continue;
          }
          if(leftSortNode!=null){
              p.next=leftSortNode;
          }else{
              p.next=rightSortNode;
          }
          break;
      }
      return dummyNode.next;
    }
~~~



**空间复杂度$O(1)$的算法实现**

~~~java
 // 自底向上归并排序
    public ListNode sortList(ListNode head) {
        if(head == null){
            return head;
        }

        // 1. 首先从头向后遍历,统计链表长度
        int length = 0; // 用于统计链表长度
        ListNode node = head;
        while(node != null){
            length++;
            node = node.next;
        }

        // 2. 初始化 引入dummynode
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        // 3. 每次将链表拆分成若干个长度为subLen的子链表 , 并按照每两个子链表一组进行合并
        for(int subLen = 1;subLen < length;subLen <<= 1){ // subLen每次左移一位（即sublen = sublen*2） PS:位运算对CPU来说效率更高
            ListNode prev = dummyHead;
            ListNode curr = dummyHead.next;     // curr用于记录拆分链表的位置

            while(curr != null){               // 如果链表没有被拆完
                // 3.1 拆分subLen长度的链表1
                ListNode head_1 = curr;        // 第一个链表的头 即 curr初始的位置
                for(int i = 1; i < subLen && curr != null && curr.next != null; i++){     // 拆分出长度为subLen的链表1
                    curr = curr.next;
                }

                // 3.2 拆分subLen长度的链表2
                ListNode head_2 = curr.next;  // 第二个链表的头  即 链表1尾部的下一个位置
                curr.next = null;             // 断开第一个链表和第二个链表的链接
                curr = head_2;                // 第二个链表头 重新赋值给curr
                for(int i = 1;i < subLen && curr != null && curr.next != null;i++){      // 再拆分出长度为subLen的链表2
                    curr = curr.next;
                }

                // 3.3 再次断开 第二个链表最后的next的链接
                ListNode next = null;        
                if(curr != null){
                    next = curr.next;   // next用于记录 拆分完两个链表的结束位置
                    curr.next = null;   // 断开连接
                }

                // 3.4 合并两个subLen长度的有序链表
                ListNode merged = mergeTwoLists(head_1,head_2);
                prev.next = merged;        // prev.next 指向排好序链表的头
                while(prev.next != null){  // while循环 将prev移动到 subLen*2 的位置后去
                    prev = prev.next;
                }
                curr = next;              // next用于记录 拆分完两个链表的结束位置
            }
        }
        // 返回新排好序的链表
        return dummyHead.next;
    }


    // 此处是Leetcode21 --> 合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1,ListNode l2){
        ListNode dummy = new ListNode(0);
        ListNode curr  = dummy;

        while(l1 != null && l2!= null){ // 退出循环的条件是走完了其中一个链表
            // 判断l1 和 l2大小
            if (l1.val < l2.val){
                // l1 小 ， curr指向l1
                curr.next = l1;
                l1 = l1.next;       // l1 向后走一位
            }else{
                // l2 小 ， curr指向l2
                curr.next = l2;
                l2 = l2.next;       // l2向后走一位
            }
            curr = curr.next;       // curr后移一位
        }

        // 退出while循环之后,比较哪个链表剩下长度更长,直接拼接在排序链表末尾
        if(l1 == null) curr.next = l2;
        if(l2 == null) curr.next = l1;

        // 最后返回合并后有序的链表
        return dummy.next; 
    }
~~~



#### [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

难度中等863收藏分享切换为英文接收动态反馈

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

**示例:**

```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

```

**说明:**

假设你总是可以到达数组的最后一个位置。

通过次数108,713提交次数281,946

**简单动态规划实现**

这道题目不同于以往的动态规划，动态规划函数不是从小到大进行推算，而是从大到小推送，定义函数$ans[j]$ 表示从第$j+1$个位置到达最后一个位置$n$的最小跳跃步数，动态规划的递推规则是从第$n-1$个元素到$1$的元素位置。动态规划递推表达式
$$
ans[i-1]=1+Math.min(ans[j])   \
其中1<=j<=Math.min(n-1,i+nums[i])
$$
代码如下

~~~java
 public int jump(int[] nums) {
        int n=nums.length;
        int[] ans=new int[n];
        ans[n-1]=0;
        int i=0,j=0;
        for(i=n-2;i>=0;i--){
            int step=nums[i];
            ans[i]=Integer.MAX_VALUE;
            for(j=i+1;j<Math.min(n,i+step+1);j++){
                if(ans[j]!=Integer.MAX_VALUE){
                    ans[i]=Math.min(ans[i],ans[j]);
                }
            }
            ans[i]=ans[i]==Integer.MAX_VALUE?Integer.MAX_VALUE:ans[i]+1;
        }
        return ans[0];
    }
~~~



**官网贪心算法实现**

~~~java
 public int jump(int[] nums) {
        int length = nums.length;
        int end = 0;
        int maxPosition = 0; 
        int steps = 0;
        for (int i = 0; i < length - 1; i++) {
            maxPosition = Math.max(maxPosition, i + nums[i]); 
            if (i == end) {
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }
~~~



#### [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

难度中等517收藏分享切换为英文接收动态反馈

给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **深拷贝**。 深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点 **。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为  `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e2.png)

```
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]

```

**示例 3：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e3.png)**

```
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]

```

**示例 4：**

```
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。

```

 

**提示：**

- `0 <= n <= 1000`
- `-10000 <= Node.val <= 10000`
- `Node.random` 为空（null）或指向链表中的节点。

通过次数63,411提交次数105,037

**巧妙解法**

该方法的难点在于如何复制$random$指针，一种比较巧妙的方法是，克隆每一对数据元素填充到对应元素的下一个节点，并插入。比如初始链表结构是

$1->3->5$，那么拷贝之后的链表结构就是$1->new1->3->new3->5->new5$,其中$new$开头的表示的是新克隆的节点，对于$random$指针，比如如果节点

$1$对应的$random$指针是$5$，那么拷贝之后的节点对应的$random$指针就是$new5$，注意判空。

~~~java
public Node copyRandomList(Node head) {
        if(head==null){
            return head;
        }
        Node cur=head,next=head.next;
        //copy and insert newly created node
        while(cur!=null){
            next=cur.next;
            Node node=new Node(cur.val);
            cur.next=node;
            node.next=next;
            cur=next;
        }
        //copy random point
        cur=head;
        while(cur!=null){
            next=cur.next;
            if(cur.random!=null){
                next.random=cur.random.next;
            }
            cur=next.next;
        }
        //split newly created node
        Node newDummyHead=new Node(0);
        Node p=newDummyHead;
        cur=head;
        while(cur!=null){
            next=cur.next;
            cur.next=next.next;
            cur=cur.next;
            p.next=next;
            p=next;
        }
        return newDummyHead.next;
    }
~~~



#### [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

难度中等293收藏分享切换为英文接收动态反馈

给定一个字符串，逐个翻转字符串中的每个单词。

**说明：**

- 无空格字符构成一个 **单词** 。
- 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
- 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。



**示例 1：**

```
输入："the sky is blue"
输出："blue is sky the"

```

**示例 2：**

```
输入："  hello world!  "
输出："world! hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。

```

**示例 3：**

```
输入："a good   example"
输出："example good a"
解释：如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

```

**示例 4：**

```
输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"

```

**示例 5：**

```
输入：s = "Alice does not even like bob"
输出："bob like even not does Alice"

```

 

**提示：**

- `1 <= s.length <= 104`
- `s` 包含英文大小写字母、数字和空格 `' '`
- `s` 中 **至少存在一个** 单词





**进阶：**

- 请尝试使用 *O*(1) 额外空间复杂度的原地解法。

通过次数122,042提交次数266,145

**传统方法实现**

~~~java
public boolean isAlphaOrDigit(int ch){
        if(ch>='a'&&ch<='z'||ch>='A'&&ch<='Z'||Character.isDigit(ch)){
            return true;
        }
        return false;
    }
    public String reverseWords(String s) {
        Stack<String> st=new Stack();
        int start=-1,end=-1;
        for(int i=0;i<s.length();i++){
            char ch=s.charAt(i);
            if(isAlphaOrDigit(ch)){
                if(start==-1){
                    start=i;
                }
                end=i;
            }else{
                if(start!=-1){
                    st.add(s.substring(start,end+1));
                    start=-1;
                    end=-1;
                }
            }
        }
        if(start!=-1){
            st.add(s.substring(start,end+1));
        }
        StringBuilder builder=new StringBuilder();
        while(!st.isEmpty()){
            String str=st.pop();
            builder.append(str);
            if(!st.isEmpty()){
                builder.append(" ");
            }
        }
        return builder.toString();
    }
~~~



$O(1)$空间复杂度解法$

通过双指针的方式，用来记录单词的前后索引位置。其中$right$指针指向单词的最后一个字母的索引位置，$left$指针指向单词的第一个字母的索引位置。

~~~java
public String reverseWords(String s) {
        int left = s.length() - 1,right = s.length() - 1;
        StringBuilder res = new StringBuilder();
        while(left >= 0){
            while(left >= 0 && s.charAt(left) != ' '){
                left--;
            }
            res.append(s.substring(left+1,right+1)).append(" ");
            while(left >= 0 && s.charAt(left) == ' '){
                left--;
            }
            right = left;
        }
        return res.toString().trim();
    }
~~~



#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

难度中等1036收藏分享切换为英文接收动态反馈

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

```

 

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` 的值为 `'0'` 或 `'1'`

通过次数217,932提交次数412,145

**我的答案**

通过dfs和bfs方法不断的去拓宽新的路径，使用$visited[m][n]$数组记录探索的路径，减少重复探索。探索的顺序应该是上下左右，**而不是单纯的右和下边**。

~~~java
public void dfs(char[][] grid,int row,int col,boolean[][] visited){
        boolean outBound=row<0||col<0||row>=visited.length||col>=visited[0].length;
        if(outBound||visited[row][col]){
            return;
        }
        visited[row][col]=true;
        if(grid[row][col]=='0'){
            return;
        }
        //up
        dfs(grid,row-1,col,visited);
        //down
        dfs(grid,row+1,col,visited);
        //left
        dfs(grid,row,col-1,visited);
        //right
        dfs(grid,row,col+1,visited);
    }
    public int numIslands(char[][] grid) {
        int m=grid.length;
        int n=grid[0].length;
        boolean[][] visited=new boolean[m][n];
        int count=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='0'||visited[i][j]){
                    continue;
                }
                count++;
                dfs(grid,i,j,visited);
            }
        }
        return count;
    }
~~~



**网上解答**

- 思路：遍历岛这个二维数组，如果当前数为1，则进入感染函数并将岛个数+1
- 感染函数：其实就是一个递归标注的过程，它会将所有相连的1都标注成2。为什么要标注？这样就避免了遍历过程中的重复计数的情况，一个岛所有的1都变成了2后，遍历的时候就不会重复遍历了。建议没想明白的同学画个图看看。



~~~java
public int numIslands(char[][] grid) {
        int islandNum = 0;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == '1'){
                    infect(grid, i, j);
                    islandNum++;
                }
            }
        }
        return islandNum;
    }
    //感染函数
    public void infect(char[][] grid, int i, int j){
        if(i < 0 || i >= grid.length ||
           j < 0 || j >= grid[0].length || grid[i][j] != '1'){
            return;
        }
        grid[i][j] = '2';
        infect(grid, i + 1, j);
        infect(grid, i - 1, j);
        infect(grid, i, j + 1);
        infect(grid, i, j - 1);
    }
~~~



#### [1115. 交替打印FooBar](https://leetcode-cn.com/problems/print-foobar-alternately/)

难度中等100收藏分享切换为英文接收动态反馈

我们提供一个类：

```
class FooBar {
  public void foo() {
    for (int i = 0; i < n; i++) {
      print("foo");
    }
  }

  public void bar() {
    for (int i = 0; i < n; i++) {
      print("bar");
    }
  }
}

```

两个不同的线程将会共用一个 `FooBar` 实例。其中一个线程将会调用 `foo()` 方法，另一个线程将会调用 `bar()` 方法。

请设计修改程序，以确保 "foobar" 被输出 n 次。

 

**示例 1:**

```
输入: n = 1
输出: "foobar"
解释: 这里有两个线程被异步启动。其中一个调用 foo() 方法, 另一个调用 bar() 方法，"foobar" 将被输出一次。

```

**示例 2:**

```
输入: n = 2
输出: "foobarfoobar"
解释: "foobar" 将被输出两次。

```

通过次数31,774提交次数56,994

**我的解法**

本质上是线程的同步问题，一个打印foo线程，一个打印bar线程。首先要确保二者的启动顺序，即确保foo线程在bar线程之前启动。没有通过加锁的方式实现，而是通过设置一个volatile类型的成员变量来做控制。然后是线程交替打印，通过$Object.wait()$和$Object.notify()$方法控制，注意一点，当打印最后一个字符的时候不要在调用$wait()$方法了

~~~java
class FooBar {
    private int n;
    private volatile boolean fooStart=false;
    public FooBar(int n) {
        this.n = n;
    }

    public void foo(Runnable printFoo) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            synchronized(this){
                if(!fooStart){
                    fooStart=true;
                }
                notify();
        	    // printFoo.run() outputs "foo". Do not change or remove this line.
        	    printFoo.run();
                if(i!=n-1){
                    wait();
                }
            }
        }
    }
    public void bar(Runnable printBar) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            synchronized(this){
                //be sure printFoo thread start before current thread
                if(!fooStart){
                    wait();
                }
                notify();
                // printBar.run() outputs "bar". Do not change or remove this line.
        	    printBar.run();
                if(i!=n-1){
                    wait();
                }
            }
        }
    }
}
~~~



$通过ReentrantLock实现$

~~~java
class FooBar {
    private int n;
    private AtomicInteger flag = new AtomicInteger(0);
    private ReentrantLock lock = new ReentrantLock();
    private Condition condition1 = lock.newCondition();
    private Condition condition2 = lock.newCondition();

    public FooBar(int n) {
        this.n = n;
    }

    public void foo(Runnable printFoo) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            lock.lock();
            try {
                while (flag.get() != 0) {
                    condition1.await();
                }

                printFoo.run();
                flag.set(1);
                condition2.signal();
            } finally {
                lock.unlock();
            }
        }
    }

    public void bar(Runnable printBar) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            lock.lock();
            try {
                while (flag.get() != 1) {
                    condition2.await();
                }

                printBar.run();
                flag.set(0);
                condition1.signal();
            } finally {
                lock.unlock();
            }
        }
    }
}


~~~



#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

难度中等1091收藏分享切换为英文接收动态反馈

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

 

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

```

 

**提示：**

- `1 <= nums.length <= 3 * 104`
- `0 <= nums[i] <= 105`

通过次数207,667提交次数496,273

**dp实现，算法时间复杂度$O(n^2)$**

定义$ans[i]$表示的是第$i$个索引下标能够到达最后一个索引，注意初始化最后一个索引该值要设置为true，即$ans[n-1]=true$。

~~~java
 public boolean canJump(int[] nums) {
        //典型的dp算法，定义ans[i]表示从数组第i个下标是否能够跳跃到最后一个下标
        int n =nums.length;
        boolean[] reach=new boolean[n];
        reach[n-1]=true;
        for(int i=n-2;i>=0;i--){
            for(int j=i+1;j<Math.min(n,i+1+nums[i]);j++){
                if(reach[j]){
                    reach[i]=true;
                    break;
                }
            }
        }
        return reach[0];
    }
~~~



**贪心算法实现，时间复杂度$O(n)$**

我们可以用贪心的方法解决这个问题。

设想一下，对于数组中的任意一个位置 $y$，我们如何判断它是否可以到达？根据题目的描述，只要存在一个位置 $x$，它本身可以到达，并且它跳跃的最大长度为 $x+nums[x]$，这个值大于等于 $y$，即 $x+nums[x]≥y$，那么位置 $y$ 也可以到达。

换句话说，对于每一个可以到达的位置 $x$，它使得 $x+1, x+2, ⋯,x+nums[x] $这些连续的位置都可以到达。

这样以来，我们依次遍历数组中的每一个位置，并实时维护 最远可以到达的位置。对于当前遍历到的位置 $x$，如果它在 最远可以到达的位置 的范围内，那么我们就可以从起点通过若干次跳跃到达该位置，因此我们可以用 $x+nums[x]$ 更新 最远可以到达的位置。

在遍历的过程中，如果 最远可以到达的位置 大于等于数组中的最后一个位置，那就说明最后一个位置可达，我们就可以直接返回 True 作为答案。反之，如果在遍历结束后，最后一个位置仍然不可达，我们就返回 False 作为答案。

以题目中的示例一


$[2, 3, 1, 1, 4]$
为例：

我们一开始在位置 $0$，可以跳跃的最大长度为 $2$，因此最远可以到达的位置被更新为 $2$；

我们遍历到位置 $1$，由于 $1≤2$，因此位置 $1$ 可达。我们用 $1$ 加上它可以跳跃的最大长度 $3$，将最远可以到达的位置更新为 $4$。由于 $4$ 大于等于最后一个位置 $4$，因此我们直接返回 True。

我们再来看看题目中的示例二


$[3, 2, 1, 0, 4]$
我们一开始在位置 $0$，可以跳跃的最大长度为 $3$，因此最远可以到达的位置被更新为 $3$；

我们遍历到位置 $1$，由于 $1≤3$，因此位置 $1$ 可达，加上它可以跳跃的最大长度 $2$ 得到 $3$，没有超过最远可以到达的位置；

位置 $2$、位置 $3$ 同理，最远可以到达的位置不会被更新；

我们遍历到位置 $4$，由于 $4>3$，因此位置 $4$ 不可达，我们也就不考虑它可以跳跃的最大长度了。

在遍历完成之后，位置 $4$ 仍然不可达，因此我们返回 False。

~~~java
 public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i = 0; i < n; ++i) {
            if (i <= rightmost) {
                rightmost = Math.max(rightmost, i + nums[i]);
                if (rightmost >= n - 1) {
                    return true;
                }
            }
        }
        return false;
    }
~~~



#### [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

难度中等448收藏分享切换为英文接收动态反馈

给定一个链表，旋转链表，将链表每个节点向右移动 *k *个位置，其中 *k *是非负数。

**示例 1:**

```
输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL

```

**示例 2:**

```
输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL
```

通过次数121,662提交次数299,446

**传统方法**

由示例代码知道，要进行数组旋转关键在于找到单链表的$head,tail,n-k+1（索引从1开始）$个元素，然后将彼此进行串联。设链表总长度为$n$,

那么实际旋转次数应该是$k\%n$。

~~~java
public ListNode rotateRight(ListNode head, int k) {
        if(head==null||head.next==null){
            return head;
        }
        //list length
        int n=0;
        //virtual head node
        ListNode dummy=new ListNode();
        dummy.next=head;
        ListNode p=dummy;
        while(p.next!=null){
            p=p.next;
            n++;
        }
        //p current point the tail node of single linked list
        ListNode tail=p;
        //real rotation
        k=k%n;
        if(k==0){
            return head;
        }
        //search the k-n+1 th node ,where linked list is split
        int i=0;
        //search the k-n th node
        p=dummy;
        while(i<n-k){
            p=p.next;
            i++;
        }
        ListNode newHead=p.next;
        tail.next=head;
        p.next=null;
        return newHead;
    }
~~~



#### [1318. 或运算的最小翻转次数](https://leetcode-cn.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/)

难度中等28收藏分享切换为英文接收动态反馈

给你三个正整数 `a`、`b` 和 `c`。

你可以对 `a` 和 `b` 的二进制表示进行位翻转操作，返回能够使按位或运算  `a` OR `b` == `c` 成立的最小翻转次数。

「位翻转操作」是指将一个数的二进制表示任何单个位上的 1 变成 0 或者 0 变成 1 。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/11/sample_3_1676.png)

```
输入：a = 2, b = 6, c = 5
输出：3
解释：翻转后 a = 1 , b = 4 , c = 5 使得 a OR b == c
```

**示例 2：**

```
输入：a = 4, b = 2, c = 7
输出：1
```

**示例 3：**

```
输入：a = 1, b = 2, c = 3
输出：0
```

 

**提示：**

- `1 <= a <= 10^9`
- `1 <= b <= 10^9`
- `1 <= c <= 10^9`

通过次数4,773

提交次数7,483

**我的解答**

我是采用传统的方法实现，$a、b、c$的二进制位转换本质上就是位运算的操作，每个位彼此之前其实是独立的。比如对于值$a=2,b=6,c=5$的数来说

~~~sh
a=0010
b=0110
c=0101
~~~

从高位到低位分别转换对应$00->0,01->1,11->0,00->1$，对应的转换次数分别是$0,0,2,1$。设置一个数组表示$a、b、c$相同二进制位对应的数需要的转换次数，比如$001对应的转换次数是1,011对应的转换次数是0,110对应的转换次数是2$。如此进行枚举（对应的是从高位到低位）。

~~~java
 public int minFlips(int a, int b, int c) {
        //索引表示由abc的二进制位构成的十进制表示的转换次数，比如abc相同的位
        //对应的值分别是1,0,0，那么只需要转换一次就行,a、b、c是从高位到低位
        int[] bitFormat=new int[]{0,1,1,0,1,0,2,0};
        String aStr=Integer.toBinaryString(a);
        String bStr=Integer.toBinaryString(b);
        String cStr=Integer.toBinaryString(c);
        int result=0;
        for(int i=aStr.length()-1,j=bStr.length()-1,k=cStr.length()-1;
            i>=0||j>=0||k>=0;){
            int num=0;
            if(i>=0){
                num+=4*(aStr.charAt(i)-'0');
                i--;
            }
            if(j>=0){
                num+=2*(bStr.charAt(j)-'0');
                j--;
            }
            if(k>=0){
                num+=cStr.charAt(k)-'0';
                k--;
            }
            result+=bitFormat[num];
        }
        return result;
    }
~~~



**官方解答：枚举+位运算**

由于在或（$OR$）运算中，二进制表示的每一位都是独立的，即修改 $a 或 b 二进制表示中的第 i$ 位，只会影响$ a OR b 中第 i$ 位的值，因此我们可以依次枚举并考虑每一位。注意到 $a、b 和 c 均小于 10^9$，它们的二进制表示最多有 30 位（包含 31 个二进制位的数最小为 $2^{30} = 1073741824$，已经大于 10^9），因此我们只需要从低位到高位枚举这 30 位即可。

设 a、b 和 c 二进制表示的第 i 位分别为 bit_a、bit_b 和 bit_c，根据 bit_c 的值，会有以下两种情况：

若 bit_c 的值为 0，那么 bit_a 和 bit_b 必须都为 0，需要的翻转次数为 bit_a + bit_b；

若 bit_c 的值为 1，那么 bit_a 和 bit_b 中至少有一个为 1，只有当它们都为 0 时，才需要 1 次翻转；

我们将每一位的翻转次数进行累加，在枚举完所有位之后，就得到了最小翻转次数。

~~~c++
 int minFlips(int a, int b, int c) {
        int ans = 0;
        for (int i = 0; i < 31; ++i) {
            int bit_a = (a >> i) & 1;
            int bit_b = (b >> i) & 1;
            int bit_c = (c >> i) & 1;
            if (bit_c == 0) {
                ans += bit_a + bit_b;
            }
            else {
                ans += (bit_a + bit_b == 0);
            }
        }
        return ans;
    }
~~~



#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

难度中等1331收藏分享切换为英文接收动态反馈

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

 

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

 

**提示：**

- `0 <= nums.length <= 100`
- `0 <= nums[i] <= 400`

通过次数256,817

提交次数534,985

**简单dp**

简单的动态规划算法，降低空间复杂度，使用$O(C)$的空间

~~~
 public int rob(int[] nums) {
        int n=nums.length;
        int[] ans=new int[3];
        if(n>0){
            ans[0]=nums[0];
        }
        if(n>1){
            ans[1]=Math.max(nums[0],nums[1]);
        }
        if(n<=2){
            return ans[n-1];
        }
        for(int i=2;i<n;i++){
            ans[2]=Math.max(nums[i]+ans[0],ans[1]);
            ans[0]=ans[1];
            ans[1]=ans[2];
        }
        return ans[2];
    }
~~~



**官方版本**

~~~java
public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int length = nums.length;
        if (length == 1) {
            return nums[0];
        }
        int first = nums[0], second = Math.max(nums[0], nums[1]);
        for (int i = 2; i < length; i++) {
            int temp = second;
            second = Math.max(first + nums[i], second);
            first = temp;
        }
        return second;
    }
~~~



#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

难度中等863收藏分享切换为英文接收动态反馈

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

**示例 2：**

```
输入：head = []
输出：[]
```

**示例 3：**

```
输入：head = [1]
输出：[1]
```

 

**提示：**

- 链表中节点的数目在范围 `[0, 100]` 内
- `0 <= Node.val <= 100`

 

**进阶：**你能在不修改链表节点值的情况下解决这个问题吗?（也就是说，仅修改节点本身。）

通过次数235,749

提交次数340,004

**递归解法**

~~~java
 public ListNode swapPairs(ListNode head) {
        //递归实现，比较简单
        if(head==null||head.next==null){
            return head;
        }
        ListNode cur=head,next=head.next;
        ListNode swapNode=swapPairs(next.next);
        next.next=cur;
        cur.next=swapNode;
        return next;
    }
~~~



**迭代算法**

~~~java
public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode temp = dummyHead;
        while (temp.next != null && temp.next.next != null) {
            ListNode node1 = temp.next;
            ListNode node2 = temp.next.next;
            temp.next = node2;
            node1.next = node2.next;
            node2.next = node1;
            temp = node1;
        }
        return dummyHead.next;
    }
~~~



#### [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

难度中等701收藏分享切换为英文接收动态反馈

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 *encoded_string* 正好重复 *k* 次。注意 *k* 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 *k* ，例如不会出现像 `3a` 或 `2[4]` 的输入。

 

**示例 1：**

```
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**示例 2：**

```
输入：s = "3[a2[c]]"
输出："accaccacc"
```

**示例 3：**

```
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

**示例 4：**

```
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

通过次数85,248

提交次数157,092

**递归实现**

题目本身不难，难点在于考虑问题的全面性，递归过程是如果碰到了嵌套字符串，形如$k[encoded_string]$这种形式，就进行递归调用，遍历字符串找到该嵌套字符串的起点和终点，可以采用栈的方式实现。

~~~
public String decodeString(String s,int start,int end){
        StringBuilder result=new StringBuilder();
        int leftBracketCount=0;
        for(int i=start;i<=end;i++){
            char ch=s.charAt(i);
            if(Character.isDigit(ch)){
                int left=0,right=0;
                int sum=ch-'0';
                i++;
                ch=s.charAt(i);
                while(Character.isDigit(ch)){
                    sum=sum*10+ch-'0';
                    i++;
                    ch=s.charAt(i);
                }
                //trace to end,i currently point [
                leftBracketCount++;
                i++;
                left=i;
                while(leftBracketCount>0){
                    ch=s.charAt(i);
                    if(ch=='['){
                        leftBracketCount++;
                    }
                    if(ch==']'){
                        leftBracketCount--;
                    }
                    i++;
                }
                //i currently point the next index of ] ,so need substract one
                i--;
                right=i-1;
                for(int j=0;j<sum;j++){
                    result.append(decodeString(s,left,right));
                }
            }else{
                result.append(ch);
            }
        }
        return result.toString();
    }
    public String decodeString(String s) {
        return decodeString(s,0,s.length()-1);
    }
~~~



**辅助栈法**

本题难点在于括号内嵌套括号，需要从内向外生成与拼接字符串，这与栈的先入后出特性对应。

**算法流程：**

1. 构建辅助栈 stack， 遍历字符串 s 中每个字符 c；

   * 当 c 为数字时，将数字字符转化为数字 multi，用于后续倍数计算；

   * 当 c 为字母时，在 res 尾部添加 c；

   * 当 c 为 [ 时，将当前 multi 和 res 入栈，并分别置空置 0:

     * 记录此 [ 前的临时结果 res 至栈，用于发现对应 ] 后的拼接操作；

     * 记录此 [ 前的倍数 multi 至栈，用于发现对应 ] 后，获取 multi × [...] 字符串。

     * 进入到新 [ 后，res 和 multi 重新记录。

	* 当 c 为 ] 时，stack 出栈，拼接字符串 res = last_res + cur_multi * res，其中:
      * last_res是上个 [ 到当前 [ 的字符串，例如 "3[a2[c]]" 中的 a；
      * cur_multi是当前 [ 到 ] 内字符串的重复倍数，例如 "3[a2[c]]" 中的 2。

2. 返回字符串 res。

~~~java
 public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();
        for(Character c : s.toCharArray()) {
            if(c == '[') {
                stack_multi.addLast(multi);
                stack_res.addLast(res.toString());
                multi = 0;
                res = new StringBuilder();
            }
            else if(c == ']') {
                StringBuilder tmp = new StringBuilder();
                int cur_multi = stack_multi.removeLast();
                for(int i = 0; i < cur_multi; i++) tmp.append(res);
                res = new StringBuilder(stack_res.removeLast() + tmp);
            }
            else if(c >= '0' && c <= '9') multi = multi * 10 + Integer.parseInt(c + "");
            else res.append(c);
        }
        return res.toString();
    }

~~~



**递归解法**

该递归方法一次性返回的是字符数组，数组首个元素表示的是嵌套字符的终止索引。

~~~java
 public String decodeString(String s) {
        return dfs(s, 0)[0];
    }
    private String[] dfs(String s, int i) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        while(i < s.length()) {
            if(s.charAt(i) >= '0' && s.charAt(i) <= '9') 
                multi = multi * 10 + Integer.parseInt(String.valueOf(s.charAt(i))); 
            else if(s.charAt(i) == '[') {
                String[] tmp = dfs(s, i + 1);
                i = Integer.parseInt(tmp[0]);
                while(multi > 0) {
                    res.append(tmp[1]);
                    multi--;
                }
            }
            else if(s.charAt(i) == ']') 
                return new String[] { String.valueOf(i), res.toString() };
            else 
                res.append(String.valueOf(s.charAt(i)));
            i++;
        }
        return new String[] { res.toString() };
    } 

~~~



#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

难度中等461收藏分享切换为英文接收动态反馈

给定一个包含了一些 `0` 和 `1` 的非空二维数组 `grid` 。

一个 **岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在水平或者竖直方向上相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 `0` 。)

 

**示例 1:**

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

对于上面这个给定矩阵应返回 `6`。注意答案不应该是 `11` ，因为岛屿只能包含水平或垂直的四个方向的 `1` 。

**示例 2:**

```
[[0,0,0,0,0,0,0,0]]
```

对于上面这个给定的矩阵, 返回 `0`。

 

**注意:** 给定的矩阵`grid` 的长度和宽度都不超过 50。

通过次数80,433

提交次数123,726

**传统解法**

遍历加递归

~~~java
 public int dfs(int[][] grid,int i,int j){
        int m=grid.length,n=grid[0].length;
        if(i<0||j<0||i>=m||j>=n){
            return 0;
        }
        if(grid[i][j]!=1){
            grid[i][j]=-1;
            return 0;
        }
        grid[i][j]=-1;
        return 1+dfs(grid,i-1,j)+dfs(grid,i+1,j)+dfs(grid,i,j-1)+dfs(grid,i,j+1);
    }
    public int maxAreaOfIsland(int[][] grid) {
        //直接在原来数组上进行更改，减少空间复杂度
        int maxSpace=0;
        int m=grid.length,n=grid[0].length;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                int ele=grid[i][j];
                if(ele!=1){
                    //indicate traversed
                    grid[i][j]=-1;
                    continue;
                }
                maxSpace=Math.max(maxSpace,dfs(grid,i,j));
            }
        }
        return maxSpace;
    }
~~~



**深度优先搜索**

使用$grid[i][j]=0$表示已经访问过该位置

~~~java
 public int maxAreaOfIsland(int[][] grid) {
        int ans = 0;
        for (int i = 0; i != grid.length; ++i) {
            for (int j = 0; j != grid[0].length; ++j) {
                ans = Math.max(ans, dfs(grid, i, j));
            }
        }
        return ans;
    }

    public int dfs(int[][] grid, int cur_i, int cur_j) {
        if (cur_i < 0 || cur_j < 0 || cur_i == grid.length || cur_j == grid[0].length || grid[cur_i][cur_j] != 1) {
            return 0;
        }
        grid[cur_i][cur_j] = 0;
        int[] di = {0, 0, 1, -1};
        int[] dj = {1, -1, 0, 0};
        int ans = 1;
        for (int index = 0; index != 4; ++index) {
            int next_i = cur_i + di[index], next_j = cur_j + dj[index];
            ans += dfs(grid, next_i, next_j);
        }
        return ans;
    }

~~~



**深度优先搜索+栈**

~~~java
public int maxAreaOfIsland(int[][] grid) {
        int ans = 0;
        for (int i = 0; i != grid.length; ++i) {
            for (int j = 0; j != grid[0].length; ++j) {
                int cur = 0;
                Deque<Integer> stacki = new LinkedList<Integer>();
                Deque<Integer> stackj = new LinkedList<Integer>();
                stacki.push(i);
                stackj.push(j);
                while (!stacki.isEmpty()) {
                    int cur_i = stacki.pop(), cur_j = stackj.pop();
                    if (cur_i < 0 || cur_j < 0 || cur_i == grid.length || cur_j == grid[0].length || grid[cur_i][cur_j] != 1) {
                        continue;
                    }
                    ++cur;
                    grid[cur_i][cur_j] = 0;
                    int[] di = {0, 0, 1, -1};
                    int[] dj = {1, -1, 0, 0};
                    for (int index = 0; index != 4; ++index) {
                        int next_i = cur_i + di[index], next_j = cur_j + dj[index];
                        stacki.push(next_i);
                        stackj.push(next_j);
                    }
                }
                ans = Math.max(ans, cur);
            }
        }
        return ans;
    }

~~~



**广度优先搜索**

~~~java
public int maxAreaOfIsland(int[][] grid) {
        int ans = 0;
        for (int i = 0; i != grid.length; ++i) {
            for (int j = 0; j != grid[0].length; ++j) {
                int cur = 0;
                Queue<Integer> queuei = new LinkedList<Integer>();
                Queue<Integer> queuej = new LinkedList<Integer>();
                queuei.offer(i);
                queuej.offer(j);
                while (!queuei.isEmpty()) {
                    int cur_i = queuei.poll(), cur_j = queuej.poll();
                    if (cur_i < 0 || cur_j < 0 || cur_i == grid.length || cur_j == grid[0].length || grid[cur_i][cur_j] != 1) {
                        continue;
                    }
                    ++cur;
                    grid[cur_i][cur_j] = 0;
                    int[] di = {0, 0, 1, -1};
                    int[] dj = {1, -1, 0, 0};
                    for (int index = 0; index != 4; ++index) {
                        int next_i = cur_i + di[index], next_j = cur_j + dj[index];
                        queuei.offer(next_i);
                        queuej.offer(next_j);
                    }
                }
                ans = Math.max(ans, cur);
            }
        }
        return ans;
    }
~~~



#### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

难度中等2307收藏分享切换为英文接收动态反馈

给你 `n` 个非负整数 `a1，a2，...，a``n`，每个数代表坐标中的一个点 `(i, ai)` 。在坐标内画 `n` 条垂直线，垂直线 `i` 的两个端点分别为 `(i, ai)` 和 `(i, 0)` 。找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

**说明：**你不能倾斜容器。

 

**示例 1：**

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

**示例 3：**

```
输入：height = [4,3,2,1,4]
输出：16
```

**示例 4：**

```
输入：height = [1,2,1]
输出：2
```

 

**提示：**

- `n = height.length`
- `2 <= n <= 3 * 104`
- `0 <= height[i] <= 3 * 104`

通过次数399,688

提交次数619,108

**传统笨方法**

一开始，想到的是传统暴力破解方法，算法时间复杂度是$O(n^2)$，代码如下。简单做了一下过滤

~~~
public int maxArea(int[] height) {
        int maxArea=0;
        for(int i=0;i<height.length-1;i++){
            if(height[i]*(height.length-1-i)<maxArea){
                continue;
            }
            for(int j=i+1;j<height.length;j++){
                maxArea=Math.max(maxArea,Math.min(height[i],height[j])*(j-i));
            }
        }
        return maxArea;
    }
~~~



**双指针方法**

使用两个指针分别指向数组的左右边界，每次计算当前容器所能容纳的大小。然后移动较小的那端，证明也比较简单。自己想下就可以了。

~~~java
public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int ans = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            ans = Math.max(ans, area);
            if (height[l] <= height[r]) {
                ++l;
            }
            else {
                --r;
            }
        }
        return ans;
    }
~~~



#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

难度中等1173收藏分享切换为英文接收动态反馈

给定不同面额的硬币 `coins` 和一个总金额 `amount`。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

你可以认为每种硬币的数量是无限的。

 

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins = [1], amount = 0
输出：0
```

**示例 4：**

```
输入：coins = [1], amount = 1
输出：1
```

**示例 5：**

```
输入：coins = [1], amount = 2
输出：2
```

 

**提示：**

- `1 <= coins.length <= 12`
- `1 <= coins[i] <= 231 - 1`
- `0 <= amount <= 104`

通过次数205,487

提交次数477,586

**传统的dp算法**

分析题目，不难得出对应的dp转移方程，定义$dp[j]$表示组成金额为$j$的硬币的最小个数，初始化$dp[j]=-1,dp[0]=0$，dp方程如下


$$
dp[j]=min(dp[j-s[i]]+1)	//其中dp[j-s[i]]不为-1，i表示硬币索引
$$



代码如下:

~~~java
public int coinChange(int[] coins, int amount) {
        int[] dp=new int[amount+1];
        Arrays.fill(dp,-1);
        Arrays.sort(coins);
        int cn=coins.length;
        dp[0]=0;
        for(int i=1;i<=amount;i++){
            int minV=Integer.MAX_VALUE;
            for(int j=0;j<cn;j++){
                if(minV==Integer.MAX_VALUE&&i<coins[j]){
                    break;
                }
                if(i>=coins[j]&&dp[i-coins[j]]!=-1){
                    minV=Math.min(minV,dp[i-coins[j]]+1);
                }
            }
            if(minV!=Integer.MAX_VALUE){
                dp[i]=minV;
            }
        }
        return dp[amount];
    }
~~~



**dfs算法实现**

实际上是dfs和剪枝算法的结合

~~~
public int coinChange(int[] coins, int amount) {
        if (amount<1||coins.length==0||coins==null)return 0;
        Arrays.sort(coins);
        int[] ans = new int[]{Integer.MAX_VALUE};
        dfs(coins,amount,coins.length-1,0,ans);
        return ans[0] ==Integer.MAX_VALUE ? -1 : ans[0];
    }

    /*
     * @param coins待选硬币面额
     * @param amount需要凑够的金额
     * @param coindex当前选择的硬币面额索引
     * @param count目前已经选的硬币数量
     * @param ans返回结果
     * @return
      */
    private void dfs(int[] coins, int amount, int coinIdx, int count, int[] ans) {
        /*
        * 整体策略
        * 如果凑过所需金额amount，即得到一个潜在答案，计算所需的最小count
        * 如果未凑够
        * （1）如果coins是最小面额，说明这个凑发不合理，剪枝
        * （2）如果(目前已选择的硬币数量 + 1) >= ans，说明继续往下凑，硬币数量不会小于ans，剪枝
        * （3）否则尝试选择面值比coin小的硬币去凑剩余的金额
        * (4) 减少面值为coin的硬币数量，进入 <1>*/
        for (int c = amount / coins[coinIdx]; c >= 0; c--) {
            int reamin=amount-c*coins[coinIdx];
            int curCount=count+c;
            if (reamin==0){
                // 已经优先用面值较大的硬币了
                // 如果用面值较小的硬币，凑出来的数量只会更多
                // 所以直接剪枝，没必要尝试减少大面值硬币的数量，用小面值的硬币去凑
                ans[0]=Math.min(ans[0],curCount);
                return;
            }
            // 已经是最小面值了，如果还凑不够amount，说明不可能凑出这个数目，直接剪枝
            if (coinIdx==0)return;
            // 继续往下凑，硬币数量不会小于ans，直接剪枝
            if (curCount+1>=ans[0])return;
            // 选择较小的面值凑够剩余的金额
            dfs(coins, reamin, coinIdx - 1, curCount, ans);
        }
    }
~~~



## 2021年4月

#### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

难度中等952收藏分享切换为英文接收动态反馈

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。**注意，`pos` 仅仅是用于标识环的情况，并不会作为参数传递到函数中。**

**说明：**不允许修改给定的链表。

**进阶：**

- 你是否可以使用 `O(1)` 空间解决此题？

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

 

**提示：**

- 链表中节点的数目范围在范围 `[0, 104]` 内
- `-105 <= Node.val <= 105`
- `pos` 的值为 `-1` 或者链表中的一个有效索引

通过次数211,380

提交次数387,655

**传统方法，直接用过set做去重**

~~~java
public ListNode detectCycle(ListNode head) {
        Set<ListNode> st=new HashSet();
        ListNode p=head;
        while(p!=null){
            if(st.contains(p)){
                return p;
            }
            st.add(p);
            p=p.next;
        }
        return null;
    }
~~~





**快慢指针**
这个方法是比较巧妙的方法，但是不容易想到，也不太容易理解，利用快慢指针判断是否有环很容易，但是判断环的入口就没有那么容易，之前说过快慢指针肯定会在环内相遇，见下图。

![image-20201027184755943](https://pic.leetcode-cn.com/1609485152-ljsLuE-file_1609485153092)

上图黄色节点为快慢指针相遇的节点，此时

快指针走的距离：**a+(b+c)n+b**

很容易理解b+c为环的长度，a为直线距离，b为绕了n圈之后又走了一段距离才相遇，所以相遇时走的总路程为a+(b+c)n+b，合并同类项得**a+(n+1)b+nc**。

慢指针走的距离：a+(b+c)m+b,m代表圈数。

然后我们设快指针得速度是慢指针的2倍,含义为相同时间内，快指针走过的距离是慢指针的2倍。

**a+(n+1)b+nc=2[a+(m+1)b+mc]整理得a+b=(n-2m)(b+c)，**那么我们可以从这个等式上面发现什么呢？b+c

为一圈的长度。也就是说a+b等于n-2m个环的长度。为了便于理解我们看一种特殊情况，当n-2m等于1，那么a+b=b+c整理得，a=c此时我们只需重新释放两个指针，一个从head释放，一个从相遇点释放，速度相同，因为a=c所以他俩必会在环入口处相遇，则求得入口节点索引。

即$a=(n-2m-1)(b+c)+c$。从头结点出发的指针显然需要走$a$步才能到达相遇环开始地方，而相遇点走到$a$步会到达哪里，代入公式得到，会走到这个位置$(n-2m-1)(b+c)+c+b$，显然会走到环节点开始地方。

算法流程：

1.设置快慢指针，快指针速度为慢指针的2倍

2.找出相遇点

3.在head处和相遇点同时释放相同速度且速度为1的指针，两指针必会在环入口处相遇



~~~java
public ListNode detectCycle(ListNode head) {
       //快慢指针
        ListNode fast = head;
        ListNode low  = head;
        //设置循环条件
        while(fast!=null&&fast.next!=null){
            fast=fast.next.next;
            low = low.next;
            //相遇
            if(fast==low){
                //设置一个新的指针，从头节点出发，慢指针速度为1，所以可以使用慢指针从相遇点出发
                ListNode newnode = head;
                while(newnode!=low){        
                    low = low.next;
                    newnode = newnode.next;
                }
                //在环入口相遇
                return low;
            }
        } 
        return null;
        
    }

~~~



#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

难度中等719收藏分享切换为英文接收动态反馈

给定一个非空的整数数组，返回其中出现频率前 k 高的元素。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

 

**提示：**

- 你可以假设给定的 *k* 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度**必须**优于 O(*n* log *n*) , *n* 是数组的大小。
- 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
- 你可以按任意顺序返回答案。

通过次数149,599

提交次数241,435

**传统做法**

首先遍历所有元素，统计元素的出现频率，创建一个优先队列，长度为$k$,采用小顶堆方式，不断的遍历频率表，并放入优先队列，为了减少队列的长度，每当长度超过$k$时候，删除最小出现的频率的数。算法时间复杂度是$n\log_2{k}$

~~~java
public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> num2FreqMap = new HashMap();
        for (int i = 0; i < nums.length; i++) {
            num2FreqMap.compute(nums[i], (key, val) -> val == null ? 1 : val + 1);
        }
        Comparator<Map.Entry<Integer, Integer>> entryComparator = 
                Comparator.comparingInt(entry -> entry.getValue());
        PriorityQueue<Map.Entry<Integer, Integer>> priorityQueue = new PriorityQueue<>(k, entryComparator);
        for (Map.Entry<Integer, Integer> entry : num2FreqMap.entrySet()) {
            priorityQueue.offer(entry);
            if (priorityQueue.size() > k) {
                //remove the smallest element
                priorityQueue.remove();
            }
        }
        List<Integer> result = priorityQueue.stream().map(
                entry -> entry.getKey()).collect(Collectors.toList());
        int[] primeResult = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            primeResult[i] = result.get(i);
        }
        return primeResult;

    }
~~~



**官方解答（注意优先队列存储的数据内容）**

~~~java
  public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        // int[] 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
        PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] m, int[] n) {
                return m[1] - n[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            if (queue.size() == k) {
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = queue.poll()[0];
        }
        return ret;
    }
~~~



**基于快速排序**

我们可以使用基于快速排序的方法，求出「出现次数数组」的前 $k$ 大的值。

在对数组 $arr[l…r]$ 做快速排序的过程中，我们首先将数组划分为两个部分 $arr[i…q−1]$ 与 $arr[q+1…j]$，并使得 $arr[i…q−1] $中的每一个值都不超过 $arr[q]$，且 $arr[q+1…j]$中的每一个值都大于 $arr[q]$。

于是，我们根据 $k$ 与左侧子数组 $arr[i…q−1]$ 的长度（为 $q−i$）的大小关系：

+ 如果 $k≤q−i$，则数组 $arr[l…r] $前 $k$ 大的值，就等于子数组 $arr[i…q−1]$ 前 $k$ 大的值。

+ 否则，数组 $arr[l…r]$ 前 $k$ 大的值，就等于左侧子数组全部元素，加上右侧子数组 $arr[q+1…j]$ 中前 $k−(q−i)$ 大的值。

原版的快速排序算法的平均时间复杂度为 $O(NlogN)$。我们的算法中，每次只需在其中的一个分支递归即可，因此算法的平均时间复杂度降为 $O(N)$。

~~~java
public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        List<int[]> values = new ArrayList<int[]>();
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            values.add(new int[]{num, count});
        }
        int[] ret = new int[k];
        qsort(values, 0, values.size() - 1, ret, 0, k);
        return ret;
    }

    public void qsort(List<int[]> values, int start, int end, int[] ret, int retIndex, int k) {
        int picked = (int) (Math.random() * (end - start + 1)) + start;
        Collections.swap(values, picked, start);
        
        int pivot = values.get(start)[1];
        int index = start;
        for (int i = start + 1; i <= end; i++) {
            if (values.get(i)[1] >= pivot) {
                Collections.swap(values, index + 1, i);
                index++;
            }
        }
        Collections.swap(values, start, index);

        if (k <= index - start) {
            qsort(values, start, index - 1, ret, retIndex, k);
        } else {
            for (int i = start; i <= index; i++) {
                ret[retIndex++] = values.get(i)[0];
            }
            if (k > index - start + 1) {
                qsort(values, index + 1, end, ret, retIndex, k - (index - start + 1));
            }
        }
    }
~~~





#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

难度中等862收藏分享切换为英文接收动态反馈

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
```

**示例 3：**

![img](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

 

**提示：**

- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` 和 `word` 仅由大小写英文字母组成

 

**进阶：**你可以使用搜索剪枝的技术来优化解决方案，使其在 `board` 更大的情况下可以更快解决问题？

通过次数159,564

提交次数358,020

**传统方法dfs**

回溯方法

~~~java
public boolean match(char[][] board,String word,boolean[][] visited,int i,int j,int k){
        if(k==word.length()){
            return true;
        }
        int m=board.length,n=board[0].length;
        if(i>=m||j>=n||i<=-1||j<=-1){
            return false;
        }
        if(word.charAt(k)!=board[i][j]||visited[i][j]){
            return false;
        }
        //if match ,update visited flag
        visited[i][j]=true;
        //up、down、left、right match
        //up
        if(match(board,word,visited,i,j-1,k+1)){
            visited[i][j]=false;
            return true;
        }
        //down
        if(match(board,word,visited,i,j+1,k+1)){
            visited[i][j]=false;
            return true;
        }
        //left
        if(match(board,word,visited,i-1,j,k+1)){
            visited[i][j]=false;
            return true;
        }
        //right
        if(match(board,word,visited,i+1,j,k+1)){
            visited[i][j]=false;
            return true;
        }
        visited[i][j]=false;
        return false;
    }
    public boolean exist(char[][] board, String word) {
        int m=board.length;
        int n=board[0].length;
        boolean[][] mem=new boolean[m][n];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(match(board,word,mem,i,j,0)){
                    return true;
                }
            }
        }
        return false;
    }
~~~



**网上添加上剪枝方法**

~~~java
 boolean[][] visit;
        int[] dx = {-1, 0, 0, 1};
        int[] dy = {0, -1, 1, 0};
    public boolean exist(char[][] board, String word) {
        int[] count = new int[128];
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                count[board[i][j]]++;
            }
        }
        for(char c:word.toCharArray()){
            if(--count[c]<0){
                return false;
            }
        }
        visit = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (!visit[i][j] && board[i][j] == word.charAt(0)) {
                    visit[i][j] = true;
                    if (dfs(board, word.substring(1), i, j)) {
                        return true;
                    }
                    visit[i][j] = false;
                }
            }
        }
        return false;
    }
    public boolean dfs(char[][] board, String word, int x, int y) {
        if (word == null || word.length() == 0) {
            return true;
        }
        for (int i = 0; i < 4; i++) {
            int currX = x + dx[i];
            int currY = y + dy[i];
            if (currX >= 0 && currX < board.length && currY >= 0 && currY < board[0].length && !visit[currX][currY] && board[currX][currY] == word.charAt(0)) {
                visit[currX][currY] = true;
                if(dfs(board, word.substring(1), currX, currY)){
                        return true;
                }
                visit[currX][currY] = false;
            }
        }
        return false;

    }
~~~



**官方解答**

~~~java
public boolean exist(char[][] board, String word) {
        int h = board.length, w = board[0].length;
        boolean[][] visited = new boolean[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                boolean flag = check(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean check(char[][] board, boolean[][] visited, int i, int j, String s, int k) {
        if (board[i][j] != s.charAt(k)) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        boolean result = false;
        for (int[] dir : directions) {
            int newi = i + dir[0], newj = j + dir[1];
            if (newi >= 0 && newi < board.length && newj >= 0 && newj < board[0].length) {
                if (!visited[newi][newj]) {
                    boolean flag = check(board, visited, newi, newj, s, k + 1);
                    if (flag) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }
~~~



#### [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

难度中等745收藏分享切换为英文接收动态反馈

给定一个包括 *n* 个整数的数组 `nums` 和 一个目标值 `target`。找出 `nums` 中的三个整数，使得它们的和与 `target` 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

 

**示例：**

```
输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
```

 

**提示：**

- `3 <= nums.length <= 10^3`
- `-10^3 <= nums[i] <= 10^3`
- `-10^4 <= target <= 10^4`

通过次数204,135

提交次数444,393

**常规方法**

基本步骤：首先将整个数组进行排序，按照排序后的数组，求得三个数的最接近目标和的方式。整体时间复杂度是$O(N^2)$。

~~~java
 public int twoSumCloseDiff(int[] nums,int left,int right,int target){
        int minDiff=target-(nums[left]+nums[right]);
        while(left<right){
            if(nums[left]+nums[right]==target){
                return 0;
            }
            if(Math.abs(minDiff)>Math.abs(nums[left]+nums[right]-target)){
                minDiff=target-(nums[left]+nums[right]);
            }
            if(nums[left]+nums[right]>target){
                right--;
            }else{
                left++;
            }
          
        }
        return minDiff;
    }
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int diff=target-(nums[0]+nums[1]+nums[2]);
        for(int i=0;i+2<nums.length;i++){
            int minClose=twoSumCloseDiff(nums,i+1,nums.length-1,target-nums[i]);
            if(Math.abs(minClose)<Math.abs(diff)){
                diff=minClose;
            }
        }
        return target-diff;
    }
~~~



**添加了剪枝方法**

~~~java
public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int result = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            int left = i + 1;
            int right = nums.length - 1;
            while (left != right) {
                int min = nums[i] + nums[left] + nums[left + 1];    
                if (target < min) {  // 最小值剪枝
                    if (Math.abs(result - target) > Math.abs(min - target))
                        result = min;
                    break;
                }
                int max = nums[i] + nums[right] + nums[right - 1];
                if (target > max) { 
                    if (Math.abs(result - target) > Math.abs(max - target))
                        result = max;
                    break;
                }
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == target)
                    return sum;
                else if (Math.abs(sum - target) < Math.abs(result - target)) {
                    result = sum;

                }
                if (sum > target) {
                    right--;
                    while (left != right && nums[right] == nums[right + 1]) // 过滤连续相同值
                        right--;
                } else {
                    left++;
                    while (left != right && nums[left] == nums[left - 1])// 过滤连续相同值
                        left++;
                }
            }

            while (i < nums.length - 2 && nums[i] == nums[i + 1]) // i 同样过滤连续值
                i++;
        }

        return result;
    }
~~~



**官方答案**

~~~java
 public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int best = 10000000;

        // 枚举 a
        for (int i = 0; i < n; ++i) {
            // 保证和上一次枚举的元素不相等
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            // 使用双指针枚举 b 和 c
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                // 如果和为 target 直接返回答案
                if (sum == target) {
                    return target;
                }
                // 根据差值的绝对值来更新答案
                if (Math.abs(sum - target) < Math.abs(best - target)) {
                    best = sum;
                }
                if (sum > target) {
                    // 如果和大于 target，移动 c 对应的指针
                    int k0 = k - 1;
                    // 移动到下一个不相等的元素
                    while (j < k0 && nums[k0] == nums[k]) {
                        --k0;
                    }
                    k = k0;
                } else {
                    // 如果和小于 target，移动 b 对应的指针
                    int j0 = j + 1;
                    // 移动到下一个不相等的元素
                    while (j0 < k && nums[j0] == nums[j]) {
                        ++j0;
                    }
                    j = j0;
                }
            }
        }
        return best;
    }

~~~



#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

难度中等930收藏分享切换为英文接收动态反馈

给定一个**非空**字符串 *s* 和一个包含**非空**单词的列表 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

通过次数134,023

提交次数269,498

**我的解答**

首次尝试使用dfs方法进行递归遍历，结果超时了。想到使用dp算法来解答，定义函数$ans[i]$表示单词$word$的第$0$到第$i-1$的字符串能够被给定的列表$wordList$匹配，得到动态规划转移方程
$$
ans[i-1+wordItem.length()]=ans[i-1]+match(word.substring(i-1,Math.min(word.length(),i-1+wordItem.length())))	\\
其中要求ans[i-1]=true并且match表达式为true，

		
$$


注意定义的动态规划结果数组的索引结构形式，为了逻辑上处理方便，$ans[i]$实际是从1开始进行处理的。

~~~java
 public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] ans=new boolean[s.length()+1];
        ans[0]=true;
        for(int i=1;i<=s.length();i++){
            if(!ans[i-1]){
                continue;
            }
            for(String word:wordDict){
                String substr=s.substring(i-1,Math.min(s.length(),i-1+word.length()));
                if(word.equals(substr)){
                    ans[i+word.length()-1]=true;
                }
            }
        }
        return ans[s.length()];
    }
~~~



**官方解答（不带剪枝方法）**

~~~java
public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

~~~



**优化方法(带剪枝方法)**

~~~java
public boolean wordBreak(String s, List<String> wordDict) {
        //dp[0]=true 第0个字符
        boolean[] dp = new boolean[s.length() + 1];
        Set<String> hashSet = new HashSet<>(wordDict.size());
        //用于剪枝
        int max = 0;
        for (String word : wordDict) {
            hashSet.add(word);
            if (word.length() > max) max = word.length();
        }
        dp[0] = true;
        //遍历1-n个字符
        for (int i = 1; i <= s.length(); i++) {
            // 前 j 个(dp[j] 已求出) + 第 j-i 的字符
                                    //j最多只需遍历max的长度
            for (int j = i - 1; j >= (i > max ? i - max : 0); j--) {
                if (dp[j] && hashSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
~~~



####  [386*. 字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/)

难度中等156收藏分享切换为英文接收动态反馈

给定一个整数 *n*, 返回从 $1 $到 $n $的字典顺序。

例如，

给定 $n =1 3$，返回 $[1,10,11,12,13,2,3,4,5,6,7,8,9]$ 。

请尽可能的优化算法的时间复杂度和空间复杂度。 输入的数据 $n $小于等于 $5,000,000$。

通过次数15,550提交次数21,265

**暴力破解**

算法时间复杂度是nlogn，效率较低

~~~java
public List<Integer> lexicalOrder(int n) {
        String[] str=new String[n];
        for(int i=0;i<n;i++){
            str[i]=String.valueOf(i+1);
        }
        Arrays.sort(str);
        List<Integer> result=new LinkedList();
        for(String s:str){
            result.add(Integer.valueOf(s));
        }
        return result;
    }
~~~

**dfs算法实现**

~~~java
public List<Integer> lexicalOrder(int n) {
        List<Integer> ret = new ArrayList<>();
        dfs(ret, 0, n);
        return ret;
    }

    private void dfs(List<Integer> ret, int curVal, int n) {
        for (int i = 0; i < 10; i++) {
            int newVal = curVal * 10 + i;
            if (newVal > n || newVal == 0) {
                continue;
            }
    
            ret.add(newVal);
            dfs(ret, newVal, n);
        }
    }
~~~



#### [872. 叶子相似的树](https://leetcode-cn.com/problems/leaf-similar-trees/)

难度简单90收藏分享切换为英文接收动态反馈

请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 *叶值序列* 。

![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/16/tree.png)

举个例子，如上图所示，给定一棵叶值序列为 `(6, 7, 4, 9, 8)` 的树。

如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 *叶相似 *的。

如果给定的两个头结点分别为 `root1` 和 `root2` 的树是叶相似的，则返回 `true`；否则返回 `false` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/03/leaf-similar-1.jpg)

```
输入：root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
输出：true

```

**示例 2：**

```
输入：root1 = [1], root2 = [1]
输出：true

```

**示例 3：**

```
输入：root1 = [1], root2 = [2]
输出：false

```

**示例 4：**

```
输入：root1 = [1,2], root2 = [2,2]
输出：true

```

**示例 5：**

![img](https://assets.leetcode.com/uploads/2020/09/03/leaf-similar-2.jpg)

```
输入：root1 = [1,2,3], root2 = [1,3,2]
输出：false

```

 

**提示：**

- 给定的两棵树可能会有 `1` 到 `200` 个结点。
- 给定的两棵树上的值介于 `0` 到 `200` 之间。

通过次数22,733提交次数36,229

**传统方法**

~~~java
public List<Integer> getTreeNode(TreeNode root){
        List<Integer> rootNodeList=new LinkedList();
        if(root==null){
            return rootNodeList;
        }
        if(root.left==null&&root.right==null){
            rootNodeList.add(root.val);
            return rootNodeList;
        }
        List<Integer> leftChildrenLeafNode=getTreeNode(root.left);
        List<Integer> rightChildrenLeafNode=getTreeNode(root.right);
        rootNodeList.addAll(leftChildrenLeafNode);
        rootNodeList.addAll(rightChildrenLeafNode);
        return rootNodeList;
    }

    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> tree1LeafNodeList=getTreeNode(root1);
        List<Integer> tree2LeafNodeList=getTreeNode(root2);
        if(tree1LeafNodeList.size()!=tree2LeafNodeList.size()){
            return false;
        }
        Iterator<Integer> tree1Ite=tree1LeafNodeList.iterator();
        Iterator<Integer> tree2Ite=tree2LeafNodeList.iterator();
        while(tree1Ite.hasNext()&&tree2Ite.hasNext()){
            Integer leaf1=tree1Ite.next();
            Integer leaf2=tree2Ite.next();
            if(leaf1!=leaf2){
                return false;
            }
        }
        return true;
    
    }
~~~

**中序遍历小递归**

~~~java
public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        ArrayList<Integer> arrOne = new ArrayList<Integer>();
        ArrayList<Integer> arrTwo=new ArrayList<Integer>();
        arrOne=order(arrOne, root1);
        arrTwo=order(arrTwo, root2);
        return arrOne.toString().equals(arrTwo.toString());
    }

    public ArrayList<Integer> order(ArrayList<Integer> arr,TreeNode node){
        if(node.left!=null){
            arr=order(arr, node.left);
        }
        if(node.left==null&&node.right==null){
            arr.add(node.val);
        }
        if(node.right!=null){
            arr=order(arr, node.right);
        }
        return arr;
    }
~~~

**其它方式**

~~~
 public boolean leafSimilar(TreeNode root1, TreeNode root2) {
       List<Integer> leafNode1=new LinkedList();
       List<Integer> leaftNode2=new LinkedList();
       dfs(root1,leafNode1);
       dfs(root2,leaftNode2);
       return Arrays.equals(leafNode1.toArray(),leaftNode2.toArray());
    }
    public void dfs(TreeNode root,List<Integer> leaftNode){
        if(root==null){
            return;
        }
        if(root.left==null&&root.right==null){
            leaftNode.add(root.val);
            return;
        }
        //保证左右孩子访问顺序，左孩子在右孩子之前访问,实际上是用来保证叶子节点的顺序
        dfs(root.left,leaftNode);
        dfs(root.right,leaftNode);
    }
~~~



#### [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)

难度中等548收藏分享切换为英文接收动态反馈

给定两个整数，被除数 `dividend` 和除数 `divisor`。将两数相除，要求不使用乘法、除法和 mod 运算符。

返回被除数 `dividend` 除以除数 `divisor` 得到的商。

整数除法的结果应当截去（`truncate`）其小数部分，例如：`truncate(8.345) = 8` 以及 `truncate(-2.7335) = -2`

 

**示例 1:**

```
输入: dividend = 10, divisor = 3
输出: 3
解释: 10/3 = truncate(3.33333..) = truncate(3) = 3
```

**示例 2:**

```
输入: dividend = 7, divisor = -3
输出: -2
解释: 7/-3 = truncate(-2.33333..) = -2
```

 

**提示：**

- 被除数和除数均为 32 位有符号整数。
- 除数不为 0。
- 假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231, 231 − 1]。本题中，如果除法结果溢出，则返回 231 − 1。

通过次数85,479

提交次数418,905

**我的答案**

注意考虑边界条件，边界条件即是除数是$Integer.MIN\_VALUE$，被除数是$-1$的情况，此时会发生数据溢出。其它情况不会发生数据溢出，为了处理上的方便，采用long类型进行转换，同时统一采用正数进行数据处理。使用map来保存被除数的指数次幂。用来进行数据累计。

~~~java
public int divide(int dividend, int divisor) {
        //use long instead to avoid overflow
        long lDividend=dividend,lDivisor=divisor;
        long posDividend=lDividend<0?-lDividend:lDividend;
        long posDivisor=lDivisor<0?-lDivisor:lDivisor;
        if(posDividend<posDivisor||posDividend==0){
            return 0;
        }
        List<long[]> powList=new ArrayList();
        long powSum=posDivisor;
        long exp=1;
        powList.add(new long[]{exp,powSum});
        while(powSum<=posDividend){
            powSum+=powSum;
            exp+=exp;
            powList.add(new long[]{exp,powSum});
        }
        long result=0L;
        for(int i=powList.size()-1;i>=0;i--){
            exp=powList.get(i)[0];
            powSum=powList.get(i)[1];
            if(posDividend>=powSum){
                posDividend-=powSum;
                result+=exp;
            }
        }
        boolean sameSignal=(dividend>0&&divisor>0)||(dividend<0&&divisor<0);
        //overflow
        if(result>Integer.MAX_VALUE&&sameSignal){
            return Integer.MAX_VALUE;
        }
        return sameSignal?(int)result:(int)(-result);
    }
~~~



**网上解法**

~~~java
public int divide(int dividend, int divisor) {
        if(dividend==0) return 0;
        if(divisor==Integer.MIN_VALUE){
            if(dividend==Integer.MIN_VALUE)
                return 1;
            else
                return 0;
        }
        int res=0;
        if(dividend==Integer.MIN_VALUE){
            if(divisor==-1)
                return Integer.MAX_VALUE;
            else if(divisor==Integer.MIN_VALUE)
                return 1;
            else if(divisor==1)
                return Integer.MIN_VALUE;
            else{
                res+=1;
                if(divisor<0){
                    dividend-=divisor;
                }
                else{
                    dividend+=divisor;
                }
            } 
        }
        boolean flag=false;
        if(dividend>0 && divisor<0){
            divisor*=-1;
            flag=true;   
        }
        else if(dividend<0 && divisor>0){
            dividend*=-1;
            flag=true;
        }
        else if(dividend<0 && divisor<0){
            dividend*=-1;
            divisor*=-1;
            //flag=true;
        }
        ////if(dividend>0 && divisor>0){
            while(dividend>=divisor){
                int s=divisor;int c=1;
                while(s<(dividend>>1)){
                    s+=s;
                    c+=c;
                }
                dividend-=s;
                res+=c;
            }
            
        //}
        return flag==false?res:res*-1;
    }
~~~



#### [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

难度中等618收藏分享切换为英文接收动态反馈

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

**示例 1:**

```
输入: num1 = "2", num2 = "3"
输出: "6"
```

**示例 2:**

```
输入: num1 = "123", num2 = "456"
输出: "56088"
```

**说明：**

1. `num1` 和 `num2` 的长度小于110。
2. `num1` 和 `num2` 只包含数字 `0-9`。
3. `num1` 和 `num2` 均不以零开头，除非是数字 0 本身。
4. **不能使用任何标准库的大数类型（比如 BigInteger）**或**直接将输入转换为整数来处理**。

通过次数136,492

提交次数305,444

**传统笨方法，通过字符串拼接方式**

代码效率较低，因为涉及到大量字符串的生成。基本想法就是类似于手动计算两个整数相乘的公式。将整数相乘进行分解，比如对于整数相乘

$123\times1532$的计算过程，首先计算$123\times2$之后，计算下一个步骤$123\times3$并进行移位，与上一步骤所得的结果进行累加。如此反复。

~~~java
 public StringBuilder add(String num1,String num2){
        StringBuilder result=new StringBuilder();
        int i=num1.length()-1,j=num2.length()-1;
        int carry=0;
        if(num1.isEmpty()||num2.isEmpty()){
            if(num1.isEmpty()){
                result.append(num2);
            }else{
                result.append(num1);
            }
            return result;
        }
        while(i>=0||j>=0){
            int sum=carry;
            if(i>=0){
                sum+=num1.charAt(i)-'0';
                i--;
            }
            if(j>=0){
                sum+=num2.charAt(j)-'0';
                j--;
            }
            carry=sum>9?1:0;
            result.insert(0,sum>9?sum-10:sum);
        }
        if(carry>0){
            result.insert(0,carry);
        }
        return result;
    }
    public StringBuilder mulSingle(String num1,char ch){
        StringBuilder result=new StringBuilder();
        int carry=0;
        StringBuilder zeroSuffix=new StringBuilder();
        int singleInt=ch-'0';
        for(int i=num1.length()-1;i>=0;i--){
            int val=(num1.charAt(i)-'0')*singleInt;
            String realVal=String.valueOf(val)+zeroSuffix;
            result=add(result.toString(),realVal);
            zeroSuffix.append('0');
        }
        return result;
    }
    public String multiply(String num1, String num2) {
        if(num1.equals("0")||num2.equals("0")){
            return "0";
        }
        StringBuilder zeroSuffix=new StringBuilder();
        StringBuilder result=new StringBuilder();
        for(int j=num2.length()-1;j>=0;j--){
            StringBuilder realVal=mulSingle(num1,num2.charAt(j)).append(zeroSuffix);
            result=add(result.toString(),realVal.toString());
            zeroSuffix.append('0');
        }
        return result.toString();
    }
~~~



**官方解答一，时间复杂度是$O(mn+n^2)$**

~~~java
public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        String ans = "0";
        int m = num1.length(), n = num2.length();
        for (int i = n - 1; i >= 0; i--) {
            StringBuffer curr = new StringBuffer();
            int add = 0;
            for (int j = n - 1; j > i; j--) {
                curr.append(0);
            }
            int y = num2.charAt(i) - '0';
            for (int j = m - 1; j >= 0; j--) {
                int x = num1.charAt(j) - '0';
                int product = x * y + add;
                curr.append(product % 10);
                add = product / 10;
            }
            if (add != 0) {
                curr.append(add % 10);
            }
            ans = addStrings(ans, curr.reverse().toString());
        }
        return ans;
    }

    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, add = 0;
        StringBuffer ans = new StringBuffer();
        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? num1.charAt(i) - '0' : 0;
            int y = j >= 0 ? num2.charAt(j) - '0' : 0;
            int result = x + y + add;
            ans.append(result % 10);
            add = result / 10;
            i--;
            j--;
        }
        ans.reverse();
        return ans.toString();
    }

~~~

官方解答2：时间复杂度是$O(mn)$

采用字符数组，而不是采用字符串累加方式，避免大量字符串的创建

~~~java
public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length(), n = num2.length();
        int[] ansArr = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            int x = num1.charAt(i) - '0';
            for (int j = n - 1; j >= 0; j--) {
                int y = num2.charAt(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for (int i = m + n - 1; i > 0; i--) {
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        StringBuffer ans = new StringBuffer();
        while (index < m + n) {
            ans.append(ansArr[index]);
            index++;
        }
        return ans.toString();
    }

~~~

**网上优秀答案**

~~~java
  public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length(), n = num2.length();
        int[] ansArr = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            int x = num1.charAt(i) - '0';
            for (int j = n - 1; j >= 0; j--) {
                int y = num2.charAt(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for (int i = m + n - 1; i > 0; i--) {
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        StringBuffer ans = new StringBuffer();
        while (index < m + n) {
            ans.append(ansArr[index]);
            index++;
        }
        return ans.toString();
    }
~~~











































































## shell

#### [193. 有效电话号码](https://leetcode-cn.com/problems/valid-phone-numbers/)

难度简单66收藏分享切换为英文接收动态反馈

给定一个包含电话号码列表（一行一个电话号码）的文本文件 `file.txt`，写一个单行 bash 脚本输出所有有效的电话号码。

你可以假设一个有效的电话号码必须满足以下两种格式： (xxx) xxx-xxxx 或 xxx-xxx-xxxx。（x 表示一个数字）

你也可以假设每行前后没有多余的空格字符。

 

**示例：**

假设 `file.txt` 内容如下：

```
987-123-4567
123 456 7890
(123) 456-7890
```

你的脚本应当输出下列有效的电话号码：

```
987-123-4567
(123) 456-7890
```

通过次数21,312

提交次数67,300

**答案**

可以使用egrep进行扩展正则匹配，也可以使用$grep -P$  PERL格式的正则匹配方法，注意特殊字符$()$需要进行转义。

注意""不要丢了，其中的空格，()是普通字符，" "不要丢了
^：表示行首，以...开始，这里表示以(xxx) 或者xxx-开始，注意空格
()：选择操作符，要么是([0-9]\{3\}) ，要么是[0-9]\{3\}-
|：或者连接操作符，表示或者
[]：单字符占位，[0-9]表示一位数字
{n}：匹配n位，[0-9]\{3\}匹配三位连续数字
$：表示行尾，结束

~~~shell
egrep  '^(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$' file.txt
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt
sed -n -r '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p' file.txt
awk '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-([0-9]{4})$/' file.txt
~~~



#### [195. 第十行](https://leetcode-cn.com/problems/tenth-line/)

难度简单86收藏分享切换为英文接收动态反馈

给定一个文本文件 `file.txt`，请只打印这个文件中的第十行。

**示例:**

假设 `file.txt` 有如下内容：

```
Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
```

你的脚本应当显示第十行：

```
Line 10
```

**说明:**
\1. 如果文件少于十行，你应当输出什么？
\2. 至少有三种不同的解法，请尝试尽可能多的方法来解题。

通过次数29,311

提交次数67,661

**答案**

~~~shell
cat file.txt |awk '{if(NR==10) printf "%s\n",$0; }'
awk 'NR==10' file.txt
sed -n 10p file.txt
awk '{if(NR==10) printf "%s\n",$0; }' file.txt
awk 'NR==10{print $0}' file.txt
~~~



#### [192. 统计词频](https://leetcode-cn.com/problems/word-frequency/)

难度中等136收藏分享切换为英文接收动态反馈

写一个 bash 脚本以统计一个文本文件 `words.txt` 中每个单词出现的频率。

为了简单起见，你可以假设：

- `words.txt`只包括小写字母和 `' '` 。
- 每个单词只由小写字母组成。
- 单词间由一个或多个空格字符分隔。

**示例:**

假设 `words.txt` 内容如下：

```
the day is sunny the the
the sunny is is
```

你的脚本应当输出（以词频降序排列）：

```
the 4
is 3
sunny 2
day 1
```

**说明:**

- 不要担心词频相同的单词的排序问题，每个单词出现的频率都是唯一的。
- 你可以使用一行 [Unix pipes](http://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO-4.html) 实现吗？

通过次数17,891

提交次数50,840

**答案**

~~~
cat words.txt |xargs -n1|sort -n|uniq -c|sort -k1 -rn|awk -F ' ' '{printf "%s %s\n",$2,$1;}'
cat words.txt | tr -s ' ' '\n'|sort|uniq -c |sort -r|awk '{print $2" "$1}'
cat words.txt |tr -s ' ' '\n' |sort|uniq -c|sort -r|awk '{print $2,$1}'
cat words.txt | tr ' ' '\n' | sed -e '/^$/d' | sort | uniq -c | sort -r | awk '{print $2, $1}'
cat words.txt | xargs -n1 | sort | uniq -c | sort -rn -k1 | awk '{print $2, $1}'
~~~



#### [194. 转置文件](https://leetcode-cn.com/problems/transpose-file/)

难度中等49收藏分享切换为英文接收动态反馈

给定一个文件 `file.txt`，转置它的内容。

你可以假设每行列数相同，并且每个字段由 `' '` 分隔。

 

**示例：**

假设 `file.txt` 文件内容如下：

```
name age
alice 21
ryan 30
```

应当输出：

```
name alice ryan
age 21 30
```

通过次数8,433

提交次数24,574

**我的答案**

直接通过bash脚本运行，注意控制空格和换行符的输出。每一行的最后一列之后不要再输出空格符，最后一行之后不要再输出换行符。echo不换行通过命令echo \-e "\c"​来表示。print会自动幻皇，printf不会换行。

~~~shell
row=$(cat file.txt|wc -l)
if [[ $row -eq 0 ]];then
        echo "empty line"
        return 0
fi
col=$(head -1 file.txt|xargs -n1|wc -l)
arr=()
rp=$(cat file.txt)
index=0
for i in ${rp[@]};do
        arr[index]=$i
        let index++;
done

for ((i=0;i<col;i++));do
        for ((j=0;j<row;j++));do
                index=$((j*col+i))
                echo -e "${arr[$index]}\c"
                if((j!=row-1));then
                        echo -e " \c"
                fi
        done
        if((i!=col-1));then
                echo
        fi
done
~~~



**awk命令解答**

~~~shell
awk '{ #这个大括号里的代码是 对正文的处理
    # NF表示列数，NR表示已读的行数
    # 注意for中的i从1开始，i前没有类型
    for (i=1; i<=NF; i++){#对每一列
        if(NR==1){       #如果是第一行
            #将第i列的值存入res[i],$i表示第i列的值，i为数组的下标，以列序号为下标，
            #数组不用定义可以直接使用
            res[i]=$i;   
        }
        else{
            #不是第一行时，将该行对应i列的值拼接到res[i]
            res[i]=res[i] " " $i
        }
    }
}
# BEGIN{} 文件进行扫描前要执行的操作；END{} 文件扫描结束后要执行的操作。
END{
    #输出数组
	for (i=1; i<=NF; i++){
		print res[i]
	}
}' file.txt
~~~



**awk简单用法**

awk后面参数如果跟的是文件，那么输出的内容是文件多少列的内容。比如文件内容如下：

~~~
name age
alice 21
ryan 30
~~~

awk  '{print $2}' file.txt输出内容就是第二列内容，xargs会把对应的换行变成一行。所以有

~~~shell
# Read from the file file.txt and print its transposed content to stdout.
# 获取第一行，然后用wc来获取列数
COLS=`head -1 file.txt | wc -w`
# 使用awk依次去输出文件的每一列的参数，然后用xargs做转置
for (( i = 1; i <= $COLS; i++ )); do
    # 这里col就是在代码里要替换的参数，而它等于$i
    awk -v col=$i '{print $col}' file.txt | xargs
done
~~~

# 

