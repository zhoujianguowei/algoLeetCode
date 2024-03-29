# leetcode

## 预备知识

### 类型转换常用方法

~~~java
	int[] data = {4,5,6,7,8};

    // int[]转List<Integer>
    List<Integer> list1 = Arrays.stream(data).boxed().collect(Collectors.toList());

    // int[]转Integer[]
    Integer[] integers1 = Arrays.stream(data).boxed().toArray(Integer[]::new);

    //List<Integer> 转int[]
    int[] arr1 = list1.stream().mapToInt(Integer::intValue).toArray();

    //List<Integer> 转Integer[]
    Integer[] integers2 = list1.toArray(new Integer[0]);

    //Integer[] 转int[]
    int[] arr2 = Arrays.stream(integers1).mapToInt(Integer::intValue).toArray();

    //Integer[] 转 List<Integer>
    List<Integer> list2 = Arrays.asList(integers1);
~~~



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

该算法实际上是递归算法展开的形式，遇到节点就访问，不断将当前节点压入栈。

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



**非递归解法二**

~~~java
private static List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode temp = stack.poll();
            res.add(temp.val);
            if (temp.right != null) {
                stack.push(temp.right);
            }
            if (temp.left != null) {
                stack.push(temp.left);
            }
        }
        return res;
    }
~~~



**非递归遍历三**

~~~java
 private enum Action {
        /**
         * 如果当前结点有孩子结点（左右孩子结点至少存在一个），执行 GO
         */
        GO,
        /**
         * 添加到结果集（真正输出这个结点）
         */
        ADDTORESULT
    }

    private class Command {
        private Action action;
        private TreeNode node;

        /**
         * 将动作类与结点类封装起来
         *
         * @param action
         * @param node
         */
        public Command(Action action, TreeNode node) {
            this.action = action;
            this.node = node;
        }
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Deque<Command> stack = new ArrayDeque<>();
        stack.addLast(new Command(Action.GO, root));
        while (!stack.isEmpty()) {
            Command command = stack.removeLast();
            if (command.action == Action.ADDTORESULT) {
                res.add(command.node.val);
            } else {
                // 特别注意：以下的顺序与递归执行的顺序反着来，即：倒过来写的结果
                // 前序遍历：根结点、左子树、右子树、
                // 添加到栈的顺序：右子树、左子树、根结点
                if (command.node.right != null) {
                    stack.add(new Command(Action.GO, command.node.right));
                }
                if (command.node.left != null) {
                    stack.add(new Command(Action.GO, command.node.left));
                }
                stack.add(new Command(Action.ADDTORESULT, command.node));
            }
        }
        return res;
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

##### 方法一：递归版本

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



##### 方法二：迭代版本

~~~java
 public boolean isValidBST(TreeNode root) {
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        double inorder = -Double.MAX_VALUE;

        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
              // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
            if (root.val <= inorder) {
                return false;
            }
            inorder = root.val;
            root = root.right;
        }
        return true;
    }

~~~



##### 方法三：区间版本

按照递归定义，确定对应节点的左右子树的边界条件。

~~~java
  public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValidBST(TreeNode node, long lower, long upper) {
        if (node == null) {
            return true;
        }
        if (node.val <= lower || node.val >= upper) {
            return false;
        }
        return isValidBST(node.left, lower, node.val) && isValidBST(node.right, node.val, upper);
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



##### **基本实现**

递归中返回-1表示子树已经不是平衡的二叉树了

~~~java
	public int getHeight(TreeNode root){
        if(root==null){
            return 0;
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



##### **优化实现（实际上是实现了减枝计算）**

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

采用后序的递归遍历算法，不太容易理解。节点需要改变的串联点是左子树的最右节点的右指针指向当前节点，该方法记忆了左孩子的最后一个访问节点。

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

- `next()` 和 `hasNext()` 操作的时间复杂度是 $O(1)$，并使用 O(*h*) 内存，其中 *h*是树的高度。
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

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的**二叉搜索树**。

 

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

##### **精简的解法**

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



##### 递归解法

从二叉搜索树的定义，自然能够联想到以下递归过程，当前节点所能够构成的二叉搜索树序列，应该由小于当前节点的二叉搜索树和大于该节点的二叉搜索树笛卡尔积组成。**注意，为了方便统一做处理，如果子树序列为空，返回空列表，列表里面有一个null**

~~~java
public List<TreeNode> generateTrees(int n) {
        int[] nums=new int[n];
        for(int i=1;i<=n;i++){
            nums[i-1]=i;
        }
        return dfs(nums,0,n-1);
    }
    public List<TreeNode> dfs(int[] nums,int left,int right){
        List<TreeNode> ans=new ArrayList();
        if(left>right){
            ans.add(null);
            return ans;
        }
        for(int i=left;i<=right;i++){
            List<TreeNode> leftChildren=dfs(nums,left,i-1);
            List<TreeNode> rightChildren=dfs(nums,i+1,right);
            for(TreeNode leftChild:leftChildren){
                for(TreeNode rightChild:rightChildren){
                    TreeNode root=new TreeNode(nums[i]);
                    root.left=leftChild;
                    root.right=rightChild;
                    ans.add(root);
                }

            }
        }
        return ans;
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

  3
  / \
  4  5
 / \
 1  2
给定的树 B：

  4 
 /
 1
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



#### [889. 根据前序和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

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



##### **我的解法**

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



##### **官方版本**

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



#### [669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

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



#### [285. 二叉搜索树中的顺序后继](https://leetcode-cn.com/problems/inorder-successor-in-bst/)

难度中等88收藏分享切换为英文接收动态反馈

给你一个二叉搜索树和其中的某一个结点，请你找出该结点在树中顺序后继的节点。

结点 `p` 的后继是值比 `p.val` 大的结点中键值最小的结点。

 

**示例 1:**

![img](http://assets.leetcode.com/uploads/2019/01/23/285_example_1.PNG)

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

##### **答案**

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



##### **非递归做法**

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

##### **题解**

刚开始没有思路，看了评论才知道如何去判断。先序遍历顺序是$root>left>right$的形式，也就是说从当前遍历节点出发，遍历整个序列查找到第一个比当前节点大的节点，那么该节点以及该节点之后的序列都是右子树，即要求该节点以及以后的节点的值都大于根节点。

双重循环暴力破解，算法时间复杂度$O(n^2)$

~~~java
public boolean verifyPreorder(int[] preorder) {
        int length=preorder.length;
        for(int i=0;i<length;i++){
            boolean bigger=false;
            for(int j=i+1;j<length;j++){
                if(bigger&&preorder[i]>preorder[j]){
                    return false;
                }
                if(preorder[i]<preorder[j]){
                    bigger=true;
                }
            }
        }
        return true;
    }
~~~



##### **方法二单调栈 **

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



##### 通过set结构形式的滑动窗口

跟上例类似，也是采用滑动窗口的算法思路，不过不是采用map结构，而是采用set结构形式。每次循环的时候，移除滑动窗口中最左边的一个元素，然后尝试不断扩展当前滑动窗口的长度。注意索引的下标以及如何删除索引。

其中索引$i->end$是滑动窗口里面的内容。

~~~java
public int lengthOfLongestSubstring(String s) {
        // 哈希集合，记录每个字符是否出现过
        Set<Character> occ = new HashSet<Character>();
        int n = s.length();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                // 不断地移动右指针
                occ.add(s.charAt(rk + 1));
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = Math.max(ans, rk - i + 1);
        }
      return ans;
    }
~~~



##### 不采用map，而是采用字符标记的方式

采用int数组来标记当前出现的重复的字符的最左下标，而不是传统的通过map的形式。

~~~java
 public int lengthOfLongestSubstring(String s) {
        int[] last = new int[128];
        int res = 0;
        int start = 0;
        for(int i = 0; i < s.length(); i++) {
            int index = s.charAt(i);
            start = Math.max(start, last[index]);
            res = Math.max(res, i - start + 1);
            last[index] = i+1;
        }
        return res;
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

该题有两种比较直接的做法，一种采用优先队列方式，采用小顶堆使用，队列大小是k，初始时刻，一直添加元素到队里，当堆大小达到k时，比较当前元素和堆根元素，只有当前元素比根元素大的时候，才将当前根元素移除，当前元素添加到堆并调整堆，遍历完成后，根元素即目标，时间复杂度是$nlogk$。对于快排来说，采用随机快排方式，时间复杂度是$o(n)$。代码如下

~~~java
 public void swapInt(int[] nums,int i,int j){
        int tmp=nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
    }
    //使用随机快排实现
    public void quickSort(int[] nums,int left,int right,int index){
        if(right<=left){
            return;
        }
        Random rand=new Random();
        int randIndex=left+rand.nextInt(right-left);
        swapInt(nums,randIndex,right);
        int j=left-1;
        for(int i=left;i<=right;i++){
            if(nums[i]<=nums[right]){
                swapInt(nums,i,++j);
            }
        }
        if(j==index){
            return;
        }else if(j>index){
            quickSort(nums,left,j-1,index);
        }else{
            quickSort(nums,j+1,right,index);
        }
    }
    public int findKthLargest(int[] nums, int k) {
        quickSort(nums, 0, nums.length-1,nums.length-k);
        return nums[nums.length-k];
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



##### 后续遍历

只要保证左子树先于右子树访问即可

~~~java
public void postOrder(TreeNode root,List<Integer> ans){
        if(root==null){
            return;
        }
        postOrder(root.left,ans);
        postOrder(root.right,ans);
        if(root.left==null&&root.right==null){
            ans.add(root.val);
        }
    }
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> r1=new ArrayList();
        List<Integer> r2=new ArrayList();
        postOrder(root1,r1);
        postOrder(root2,r2);
        if(r1.size()!=r2.size()){
            return false;
        }
        for(int i=0;i<r1.size();i++){
            if(r1.get(i)!=r2.get(i)){
                return false;
            }
        }
        return true;
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
            //移位个数,注意计算结果，curr是每一轮的计算结果的逆序
            //从左到右是按照低位到高位的顺序，累计的时候要进行逆序
            //这一个内循环耗时为N^2
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







### 2021年8月

#### [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

难度简单1712

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

 

**示例 1：**

```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**示例 2：**

```
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
```

 

**提示：**

- `0 <= strs.length <= 200`
- `0 <= strs[i].length <= 200`
- `strs[i]` 仅由小写英文字母组成

通过次数578,059

提交次数1,429,842

**我的解答**

第一个没有看清题目，以为是公共最长字符串长度，仔细看是公共最长前缀。递归方式实现最简单，递归函数返回的是最长的匹配前缀字符串长度。也可以考虑用迭代，效果一样。

~~~java
 public int getLongestMaxLength(String[] strs,int start){
        if(strs.length==0||strs[0].length()<=start){
            return 0;
        }
        boolean matched=true;
        char target=strs[0].charAt(start);
        for(String str:strs){
            if(start>=str.length()||str.charAt(start)!=target){
                matched=false;
                break;
            }
        }
        if(!matched){
            return 0;
        }
        return 1+getLongestMaxLength(strs,start+1);
    }
    public String longestCommonPrefix(String[] strs) {
       int maxLength=getLongestMaxLength(strs,0);
       if(maxLength==0){
           return "";
       }
       return strs[0].substring(0,maxLength);
    }
~~~



**官方迭代做法**

~~~java
public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        int length = strs[0].length();
        int count = strs.length;
        for (int i = 0; i < length; i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < count; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }
~~~



#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

难度中等1263

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

 

**提示：**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中的所有元素 **互不相同**

通过次数282,926

提交次数353,710

**我的解法**

传统直接解法，考虑子问题，从$n$个不同元素的数组中取出$k$个不同的元素的组合（组合没有顺序要求），为了保证组合唯一，采用有序方法。

~~~java
  public List<List<Integer>> subsets(int[] nums) {
        List<Integer> numList = Arrays.stream(nums).boxed().collect(Collectors.toList());
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i <= nums.length; i++) {
            List<List<Integer>> kAnsList = new ArrayList<>();
            pickK(numList, i, new LinkedList<>(), kAnsList, 0);
            ans.addAll(kAnsList);
        }
        return ans;
    }
	/**
	** 其中i代表取出多少个元素，colNum表示暂存的元素，start表示的是开始的元素索引
	**/
    private void pickK(List<Integer> numList, final int i, Deque<Integer> colNum, List<List<Integer>> kAnsList, int start) {
        if (i == colNum.size()) {
            kAnsList.add(new ArrayList<>(colNum));
            return;
        }
        int remainNum=i-colNum.size();
        for (int j = start; j+remainNum<=numList.size(); j++) {
            colNum.add(numList.get(j));
            //保证顺序，索引顺序采用增量方式
            pickK(numList, i, colNum, kAnsList, j + 1);
            colNum.removeLast();
        }
    }
~~~



**官方解答**

记原序列中元素的总数为 $n$。原序列中的每个数字 $a_i$ 的状态可能有两种，即「在子集中」和「不在子集中」。我们用 $1$ 表示「在子集中」，$0$ 表示不在子集中，那么每一个子集可以对应一个长度为 $n$ 的 $0/1$序列，第 $i$ 位表示 $a_i$ 是否在子集中。于是一共有$2^n$中状态，时间复杂度是$n*2^n$。

~~~java
List<Integer> t = new ArrayList<Integer>();
    List<List<Integer>> ans = new ArrayList<List<Integer>>();

    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        for (int mask = 0; mask < (1 << n); ++mask) {
            t.clear();
            for (int i = 0; i < n; ++i) {
                if ((mask & (1 << i)) != 0) {
                    t.add(nums[i]);
                }
            }
            ans.add(new ArrayList<Integer>(t));
        }
        return ans;
    }
~~~

**dfs做法**

根据是否选择当前索引所在元素来进行dfs遍历。

~~~java
 	List<Integer> t = new ArrayList<Integer>();
    List<List<Integer>> ans = new ArrayList<List<Integer>>();

    public List<List<Integer>> subsets(int[] nums) {
        dfs(0, nums);
        return ans;
    }

    public void dfs(int cur, int[] nums) {
        if (cur == nums.length) {
            ans.add(new ArrayList<Integer>(t));
            return;
        }
        //选择当前元素nums[cur]
        t.add(nums[cur]);
        dfs(cur + 1, nums);
        t.remove(t.size() - 1);
        //不选择当前元素
        dfs(cur + 1, nums);
    }
~~~



#### [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

难度中等1042

给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

**示例 1 :**

```
输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。
```

**说明 :**

1. 数组的长度为 [1, 20,000]。
2. 数组中元素的范围是 [-1000, 1000] ，且整数 **k** 的范围是 [-1e7, 1e7]。

通过次数134,756

提交次数301,763

**传统直接算法**

直接二次循环遍历，计算索引$i$到索引$j$的和，时间复杂度是$O(n^2)$。

~~~java
public int subarraySum(int[] nums, int k) {
        int count=0;
        for(int i=0;i<nums.length;i++){
            int sum=0;
            for(int j=i;j<nums.length;j++){
                sum+=nums[j];
                if(sum==k){
                    count++;
                }
            }
        }
        return count;
    }
~~~



**前缀和和hash表**

我们可以基于方法一利用数据结构进行进一步的优化，我们知道方法一的瓶颈在于对每个 $i$，我们需要枚举所有的 $j$ 来判断是否符合条件，这一步是否可以优化呢？答案是可以的。

我们定义 $pre[i]$为$[0..i]$ 里所有数的和，则 $pre[i]$ 可以由 $pre[i−1]$ 递推而来，即：

$$
pre[i]−pre[j−1]==k
$$




简单移项可得符合条件的下标 $j$ 需要满足


$$
pre[j−1]==pre[i]−k
$$
所以我们考虑以 $i$ 结尾的和为 $k$ 的连续子数组个数时只要统计有多少个前缀和为 $pre[i]−k$ 的 $pre[j]$ 即可。我们建立哈希表 $mp$，以和为键，出现次数为对应的值，记录 $pre[i]$ 出现的次数，从左往右边更新 $mp$ 边计算答案，那么以 $i$ 结尾的答案 $mp[pre[i]−k]$ 即可在$O(1)$ 时间内得到。最后的答案即为所有下标结尾的和为 $k$ 的子数组个数之和。

需要注意的是，从左往右边更新边计算的时候已经保证了$mp[pre[i]−k]$ 里记录的 $pre[j]$ 的下标范围是 $ 0≤j≤i$ 。同时，由于$pre[i]$ 的计算只与前一项的答案有关，因此我们可以不用建立 $pre$ 数组，直接用 $pre$ 变量来记录$pre[i−1]$ 的答案即可。

~~~java
public int subarraySum(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap < Integer, Integer > mp = new HashMap < > ();
        mp.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (mp.containsKey(pre - k)) {
                count += mp.get(pre - k);
            }
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;
    }
~~~



#### [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

难度简单441

给定一个整数数组，判断是否存在重复元素。

如果存在一值在数组中出现至少两次，函数返回 `true` 。如果数组中每个元素都不相同，则返回 `false` 。

 

**示例 1:**

```
输入: [1,2,3,1]
输出: true
```

**示例 2:**

```
输入: [1,2,3,4]
输出: false
```

**示例 3:**

```
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
```

通过次数324,881

提交次数578,959

**常规做法**

~~~java
public boolean containsDuplicate(int[] nums) {
       Set<Integer> set=new HashSet();
       for(int num:nums){
           if(set.contains(num)){
               return true;
           }
           set.add(num);
       }
       return false;
    }
~~~



**自定义hash算法**

时间复杂度严格在$o(n)$级别。

~~~java
 int[] hash = null;
    int N = 0;
    public boolean containsDuplicate(int[] nums) {
        hash = new int[nums.length * 2];
        N = hash.length;
        Arrays.fill(hash,-1);
        for(int i = 0; i < nums.length; i++){
            if(contains(nums[i])){
                return true;
            }
        }
        return false;
    }

    public boolean contains(int value){
        int n = ((value % N) + N) % N;
        while(hash[n] != -1){
            if(hash[n] == value){
                return true;
            }
            n++;
        }
        hash[n] = value;
        return false;
    }
~~~



#### [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

难度简单789

给你一个含 `n` 个整数的数组 `nums` ，其中 `nums[i]` 在区间 `[1, n]` 内。请你找出所有在 `[1, n]` 范围内但没有出现在 `nums` 中的数字，并以数组的形式返回结果。

 

**示例 1：**

```
输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6]
```

**示例 2：**

```
输入：nums = [1,1]
输出：[2]
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 105`
- `1 <= nums[i] <= n`

**进阶：**你能在不使用额外空间且时间复杂度为 `O(n)` 的情况下解决这个问题吗? 你可以假定返回的数组不算在额外空间内。

通过次数127,790

提交次数197,737

**我的解法**

题目要求不使用额外的空间复杂度，又已知条件所有数字在$[1,n]$之内，联想到使用位置填充法，即使将数字$j(1<=j<=n)$填充到数组索引位置为$j-1$处。比如对于数组$4,2,2,3$；下面是转换的步骤

~~~javascript
//初始数组
4,2,2,3
//索引为0，遇到数字4，对应位置为3（位置3的数字是3）。将4和位置为3的数字交换得到
3,2,2,4
//索引为0，遇到数字3，对应位置2（位置2的数字是2）。将3与位置2的数字交换得到
2,2,3,4
//索引0，看位置1处所在数是否是2，如果不是进行交换，位置1处为2，不交换，前进1步
2,2,3,4
//索引1，数字2，位置正确，不交换，前进1步
2,2,3,4
//索引2，数字3，不交换，前进1步
2,2,3,4
//索引3，数字4，不交换，前进1步，结束
~~~

交换完成后，进行数组遍历，位置$j$处，如果不满足$num[j]=j+1$，那么说明该位置缺少数组$j+1$。时间复杂度是$o(n)$。

~~~java
 public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> ans=new ArrayList();
        for(int i=0;i<nums.length;i++){
            int j=nums[i];
            //把当前元素放到正确的位置，nums[nums[i]-1]
            while(nums[j-1]!=j){
                nums[i]=nums[j-1];
                nums[j-1]=j;
                j=nums[i];
            }
        }
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=i+1){
                ans.add(i+1);
            }
        }
        return ans;
    }
~~~



**官方解答**

官方解答的思想不是通过将遇到的数交换到对应的位置上，而是将遇到的数所对应的位置上进行标记（对应位置所在数+n）。比如数组$2,2,4,1$。对于位置$0$处的数字$2$对应的位置是$1$,将位置$1$所在的数进行标记，即+4。所以标记后的数组为$2,6,4,1$。标记完成之后，进行数字遍历，如果遇到的数小于n，那么这个位置对应的数缺失，注意数字溢出问题。

~~~java
public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1) % n;
            if(nums[x]<=n){
                //进行标记
                nums[x]+=n
            }
        }
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                ret.add(i + 1);
            }
        }
        return ret;
    }
~~~



#### [692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)

难度中等372

给一非空的单词列表，返回前 *k* 个出现次数最多的单词。

返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

**示例 1：**

```
输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
    注意，按字母顺序 "i" 在 "love" 之前。
```

 

**示例 2：**

```
输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
输出: ["the", "is", "sunny", "day"]
解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，
    出现次数依次为 4, 3, 2 和 1 次。
```

 

**注意：**

1. 假定 *k* 总为有效值， 1 ≤ *k* ≤ 集合元素数。
2. 输入的单词均由小写字母组成。

 

**扩展练习：**

1. 尝试以 *O*(*n* log *k*) 时间复杂度和 *O*(*n*) 空间复杂度解决。

通过次数63,863

提交次数109,678

**hash和小顶堆**

注意小顶堆比较器的比较顺序

~~~java
 public List<String> topKFrequent(String[] words, int k) {
        //先使用hash，根据单次分布进行数据统计，然后使用优先级队列进行排序
        Map<String, AtomicInteger> map = new HashMap<>();
        for (String word : words) {
            map.computeIfAbsent(word, wd -> new AtomicInteger()).incrementAndGet();
        }
        //比较器，首先按照统计词频从小到大排序，词频相同情况下，比较单词字母顺序
        Comparator<Map.Entry<String, AtomicInteger>> comparator = ((o1, o2) -> {
            if (o1.getValue().get() != o2.getValue().get()) {
                return o1.getValue().get() - o2.getValue().get();
            }
            return o2.getKey().compareTo(o1.getKey());
        });
        PriorityQueue<Map.Entry<String, AtomicInteger>> priorityQueue = new PriorityQueue<>(k, comparator);
        for (Map.Entry<String, AtomicInteger> entry : map.entrySet()) {
            if (priorityQueue.size() < k) {
                priorityQueue.offer(entry);
                continue;
            }
            //堆顶元素比当前元素小，出堆
            if (comparator.compare(priorityQueue.peek(), entry) < 0) {
                priorityQueue.poll();
                priorityQueue.offer(entry);
            }
        }
        LinkedList<String> ans = new LinkedList<>();
        while(!priorityQueue.isEmpty()){
            Map.Entry<String,AtomicInteger> entry=priorityQueue.poll();
            ans.addFirst(entry.getKey());
        }
        return ans;
    }
~~~



#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

难度简单11811

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

 

**示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

**示例 2：**

```
输入：nums = [3,2,4], target = 6
输出：[1,2]
```

**示例 3：**

```
输入：nums = [3,3], target = 6
输出：[0,1]
```

 

**提示：**

- `2 <= nums.length <= 104`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`
- **只会存在一个有效答案**

**进阶：**你可以想出一个时间复杂度小于 `O(n2)` 的算法吗？

通过次数2,352,779

提交次数4,544,013

**简单做法**

通过map结构，对应的key表示为$target-sum$，对应的val存储的是数组对应的索引位置。比如数组$[2,11,7]$,对应的target为$9$，那么对应的map结构内容是$[9,0]、[-2,1]、[2,2]$。通过一次map就可以过滤出来目标值。

~~~java
 public int[] twoSum(int[] nums, int target) {
      Map<Integer,Integer> remain2IndexMap=new HashMap();
      int[] ans=new int[2];
      int i=0;
      for(int num:nums){
          if(remain2IndexMap.containsKey(num)){
              ans[0]=i;
              ans[1]=remain2IndexMap.get(num);
              break;
          }
          remain2IndexMap.put(target-num,i++);
      }
      return ans;
    }
~~~



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

难度简单1836

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)

```
输入：head = [1,2]
输出：[2,1]
```

**示例 3：**

```
输入：head = []
输出：[]
```

 

**提示：**

- 链表中节点的数目范围是 `[0, 5000]`
- `-5000 <= Node.val <= 5000`

 

**进阶：**链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？

通过次数603,809

提交次数839,463

**递归方法**

实际上有多种递归方法，传统的递归方法按照直觉思考，返回的应该是反转后的单链表，但是考虑到下面情形

$1->2->3$,假设递归到达节点1，那么按照刚才的思考，内部递归返回的链表形式应该是$3->2$，此时考虑到还应该反转$1-2$，即让1节点之后的反转链表的尾指针指向当前节点1，所以还应该返回尾结点，当然这是一种常规想法。更直接的方法，不操作反转后的链表，直接在当前递归函数中做反转即可。即保留next节点，一次递归反转节点p以及p.next，代码如下：

~~~java
 public ListNode reverseList(ListNode head) {
        if(head==null||head.next==null){
            return head;
        }
        //递归方式，实际上该递归函数返回的结果没有在递归函数
        //中使用到，反转实在递归方法中实现的  
        ListNode next=head.next;
        ListNode newHead=reverseList(next);
        next.next=head;
        head.next=null;
        return newHead;
    }
~~~

**迭代方法**

其实就是简单的头插法。

~~~java
 public ListNode reverseList(ListNode head) {
        ListNode p=head,next,newHead=null;
        while(p!=null){
            next=p.next;
            p.next=newHead;
            newHead=p;
            p=next;
        }
        return newHead;
    }
~~~



#### [7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

难度简单2921

给你一个 32 位的有符号整数 `x` ，返回将 `x` 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 `[−231, 231 − 1]` ，就返回 0。

**假设环境不允许存储 64 位整数（有符号或无符号）。**

 

**示例 1：**

```
输入：x = 123
输出：321
```

**示例 2：**

```
输入：x = -123
输出：-321
```

**示例 3：**

```
输入：x = 120
输出：21
```

**示例 4：**

```
输入：x = 0
输出：0
```

 

**提示：**

- `-231 <= x <= 231 - 1`

通过次数757,691

提交次数2,138,371

**常规解法**

直观的解法，使用一个队列依次从低位到高位收集整数的各个位数，然后进行累加合并，注意溢出的策略。

当然也可以不用队列做输出存储，在弹出末尾元素的时候直接进行计算。

~~~java
 public int reverse(int x) {
        Queue<Integer> q=new LinkedList();
        boolean isNeg=x<0;
        while(Math.abs(x)>0){
            q.offer(x%10);
            x/=10;
        }
        int ans=0;
        while(!q.isEmpty()){
            int val=q.poll();
            if(!isNeg&&ans>(Integer.MAX_VALUE-val)/10){
                return 0;
            }
            if(isNeg&&ans<(Integer.MIN_VALUE-val)/10){
                return 0;
            }
            ans=ans*10+val;
        }
        return ans;
    }
~~~



#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

难度简单2531

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

 

**示例 1：**

```
输入：s = "()"
输出：true
```

**示例 2：**

```
输入：s = "()[]{}"
输出：true
```

**示例 3：**

```
输入：s = "(]"
输出：false
```

**示例 4：**

```
输入：s = "([)]"
输出：false
```

**示例 5：**

```
输入：s = "{[]}"
输出：true
```

 

**提示：**

- `1 <= s.length <= 104`
- `s` 仅由括号 `'()[]{}'` 组成

通过次数712,350

提交次数1,604,950

**基本做法**

题目比较简单，使用栈这种数据结构，当遇到(、[、{进行入栈，当遇到）、]、}进行出栈。

~~~java
public boolean isValid(String s) {
        Stack<Character> st=new Stack();
        for(int i=0;i<s.length();i++){
            char ch=s.charAt(i);
            if(ch=='('||ch=='{'||ch=='['){
                st.push(ch);
            }else{
                if(st.isEmpty()){
                    return false;
                }
                char top=st.pop();
                switch(String.valueOf(ch)){
                    case ")":
                        if(top!='('){
                            return false;
                        }
                        break;
                    case "}":
                        if(top!='{'){
                            return false;
                        }
                        break;
                    case "]":
                        if(top!='['){
                            return false;
                        }
                        break;
                }
            }
        }
        return st.isEmpty()?true:false;
    }
~~~

**官方解答（不使用括号匹配）**

~~~java
class Solution {
    public boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }

        Map<Character, Character> pairs = new HashMap<Character, Character>() {{
            put(')', '(');
            put(']', '[');
            put('}', '{');
        }};
        Deque<Character> stack = new LinkedList<Character>();
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (pairs.containsKey(ch)) {
                if (stack.isEmpty() || stack.peek() != pairs.get(ch)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }
}
~~~



#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

难度简单1825

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**示例 2：**

```
输入：l1 = [], l2 = []
输出：[]
```

**示例 3：**

```
输入：l1 = [], l2 = [0]
输出：[0]
```

 

**提示：**

- 两个链表的节点数目范围是 `[0, 50]`
- `-100 <= Node.val <= 100`
- `l1` 和 `l2` 均按 **非递减顺序** 排列

通过次数644,137

提交次数968,901

**不使用额外空间**

直接将这两个单链表串联起来，为了避免头结点的判断，增加了一个虚拟头结点。

~~~java
 public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead=new ListNode();
        ListNode p=dummyHead,next;
        while(l1!=null||l2!=null){
            if(l1!=null&&l2!=null){
                if(l1.val<=l2.val){
                    next=l1;
                    l1=l1.next;
                }else{
                    next=l2;
                    l2=l2.next;
                }
            }else if(l1==null){
                next=l2;
                l2=l2.next;
            }else{
                next=l1;
                l1=l1.next;
            }
            p.next=next;
            p=next;
        }
        return dummyHead.next;
    }
~~~

**官网递归和迭代两种解法**

**递归解法**

~~~java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        } else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }

    }
~~~

**迭代解法：代码更精简**

~~~java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);

        ListNode prev = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 == null ? l2 : l1;

        return prehead.next;
    }
~~~

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

难度简单3507

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

 

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**示例 2：**

```
输入：nums = [1]
输出：1
```

**示例 3：**

```
输入：nums = [0]
输出：0
```

**示例 4：**

```
输入：nums = [-1]
输出：-1
```

**示例 5：**

```
输入：nums = [-100000]
输出：-100000
```

 

**提示：**

- `1 <= nums.length <= 3 * 104`
- `-105 <= nums[i] <= 105`

 

**进阶：**如果你已经实现复杂度为 `O(n)` 的解法，尝试使用更为精妙的 **分治法** 求解。

通过次数594,864

提交次数1,084,809

**dp算法**

dp算法结合滚动数组，不使用$O(N)$的空间。

~~~java
public int maxSubArray(int[] nums) {
        int ans=nums[0];
        int sum=nums[0];
        for(int i=1;i<nums.length;i++){
            sum=Math.max(nums[i],nums[i]+sum);
            ans=Math.max(sum,ans);
        }
        return ans;
    }
~~~



#### [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

难度困难1167

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

 

**示例 1：**

```
输入：nums = [1,2,0]
输出：3
```

**示例 2：**

```
输入：nums = [3,4,-1,1]
输出：2
```

**示例 3：**

```
输入：nums = [7,8,9,11,12]
输出：1
```

 

**提示：**

- `1 <= nums.length <= 5 * 105`
- `-231 <= nums[i] <= 231 - 1`

通过次数152,029

提交次数364,128

**原地交换**

基本原则就是将数组中的数值$k(1<=k<=n)$放置到索引$k-1$位置上。

~~~java
 public int firstMissingPositive(int[] nums) {
        for(int i=0;i<nums.length;i++){
            int j=nums[i];
            //将指定的数据添加到指定的位置上，比如数组3，-1,2,1
            //第一个元素应该放到索引为2的位置上得到序列2,-1,3,1
            //由于2还在范围1-4之间，继续交换得到-1,2,3,1
            //最终交互遍历得到顺序1,2,3,-1
            while(j>=1&&j<=nums.length&&j!=nums[j-1]){
                nums[i]=nums[j-1];
                nums[j-1]=j;
                j=nums[i];
            }
        }
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=i+1){
                return i+1;
            }
        }
        return nums.length+1;
    }
~~~



#### [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

难度简单1568

给你一个整数 `x` ，如果 `x` 是一个回文整数，返回 `true` ；否则，返回 `false` 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，`121` 是回文，而 `123` 不是。

 

**示例 1：**

```
输入：x = 121
输出：true
```

**示例 2：**

```
输入：x = -121
输出：false
解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```

**示例 3：**

```
输入：x = 10
输出：false
解释：从右向左读, 为 01 。因此它不是一个回文数。
```

**示例 4：**

```
输入：x = -101
输出：false
```

 

**提示：**

- `-231 <= x <= 231 - 1`

 

**进阶：**你能不将整数转为字符串来解决这个问题吗？

通过次数717,002

提交次数1,223,652

**直接接法**

题目思路比较简单，因为题目要求不能通过将整数转换成字符串形式，可以换个思路，将这个整数进行从高位到低位反转，比如对于数字$123$反转之后的值是$321$。如果一个数是回文数，那么这两个值应该是相同的。注意数值溢出问题，为了处理简单，直接使用long类型保存反转后的整数。

~~~java
 public boolean isPalindrome(int x) {
        if(x<0){
            return false;
        }
        long reverseX=0L;
        int n=x;
        while(n!=0){
            int mod=n%10;
            reverseX=reverseX*10+mod;
            n/=10;
        }
        return x==reverseX;
    }
~~~

**官方解答**

官方解答，思路跟上文一致，不过不是将整个整数进行反转，而是只反转一般（向上取整），注意反转一半的判断终止条件。

~~~java
public boolean isPalindrome(int x) {
        // 特殊情况：
        // 如上所述，当 x < 0 时，x 不是回文数。
        // 同样地，如果数字的最后一位是 0，为了使该数字为回文，
        // 则其第一位数字也应该是 0
        // 只有 0 满足这一属性
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }

        int revertedNumber = 0;
        while (x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }

        // 当数字长度为奇数时，我们可以通过 revertedNumber/10 去除处于中位的数字。
        // 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
        // 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
        return x == revertedNumber || x == revertedNumber / 10;
    }
~~~



#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

难度困难1239

给你一个链表，每 *k* 个节点一组进行翻转，请你返回翻转后的链表。

*k* 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 *k* 的整数倍，那么请将最后剩余的节点保持原有顺序。

**进阶：**

- 你可以设计一个只使用常数额外空间的算法来解决此问题吗？
- **你不能只是单纯的改变节点内部的值**，而是需要实际进行节点交换。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)

```
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex2.jpg)

```
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

**示例 3：**

```
输入：head = [1,2,3,4,5], k = 1
输出：[1,2,3,4,5]
```

**示例 4：**

```
输入：head = [1], k = 1
输出：[1]
```



**提示：**

- 列表中节点的数量在范围 `sz` 内
- `1 <= sz <= 5000`
- `0 <= Node.val <= 1000`
- `1 <= k <= sz`

通过次数207,599

提交次数316,999

**我的解答**

迭代方法，每次反转的时候，记得当前反转链表的表头、表尾以及当前访问节点所在位置。

逻辑处理比较繁琐，采用两次遍历。第一次遍历，统计当前所有节点数目，从而换算出需要反转多少次。第二次遍历是用来进行实际的链表反转。

~~~java
public ListNode reverseKGroup(ListNode head, int k) {
        ListNode reverseHead = null, reverseTail = null;
        ListNode newHead = null;
        int n = 0, totalReverseRound = 0;
        ListNode p = head, next = null;
        while (p != null) {
            n++;
            p = p.next;
        }
        //一共需要逆转多少轮
        totalReverseRound = n / k;
        int reverseIndex = 0, reverseRound = 0;
        p = head;
        ListNode curP = null;
        ListNode preReverseTail = null;
        while (reverseRound < totalReverseRound) {
            curP = null;
            reverseIndex = 0;
            //进行一轮的逆转
            while (p != null && reverseIndex < k) {
                next = p.next;
                if (reverseIndex == 0) {
                    reverseTail = p;
                    reverseTail.next=null;
                }
                p.next = curP;
                curP = p;
                reverseIndex++;
                //该轮逆转的最后一个节点
                if (reverseIndex == k) {
                    reverseHead = p;
                    reverseRound++;
                    //逆转的新链表表头
                    if (reverseRound == 1) {
                        newHead = reverseHead;
                    }
                    if (preReverseTail != null) {
                        preReverseTail.next = reverseHead;
                    }
                    preReverseTail = reverseTail;
                }
                p=next;
            }
            if (reverseRound == totalReverseRound) {
                //keep remain linked list and break;
                reverseTail.next = next;
            }
        }
        return newHead;
    }
~~~



**官方解答**

~~~java
public ListNode reverseKGroup(ListNode head, int k) {
        ListNode hair = new ListNode(0);
        hair.next = head;
        ListNode pre = hair;

        while (head != null) {
            ListNode tail = pre;
            // 查看剩余部分长度是否大于等于 k
            for (int i = 0; i < k; ++i) {
                tail = tail.next;
                if (tail == null) {
                    return hair.next;
                }
            }
            ListNode nex = tail.next;
            ListNode[] reverse = myReverse(head, tail);
            head = reverse[0];
            tail = reverse[1];
            // 把子链表重新接回原链表
            pre.next = head;
            tail.next = nex;
            pre = tail;
            head = tail.next;
        }

        return hair.next;
    }

    public ListNode[] myReverse(ListNode head, ListNode tail) {
        ListNode prev = tail.next;
        ListNode p = head;
        while (prev != tail) {
            ListNode nex = p.next;
            p.next = prev;
            prev = p;
            p = nex;
        }
        return new ListNode[]{tail, head};
    }
~~~



#### [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

难度简单2153

给你一个有序数组 `nums` ，请你**[ 原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

 

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

**示例 1：**

```
输入：nums = [1,1,2]
输出：2, nums = [1,2]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
```

**示例 2：**

```
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
```

 

**提示：**

- `0 <= nums.length <= 3 * 104`
- `-104 <= nums[i] <= 104`
- `nums` 已按升序排列

 

通过次数758,954

提交次数1,404,961

**经典快慢指针问题**

~~~java
   public int removeDuplicates(int[] nums) {
        //使用双重指针，一个慢指针指示新数组，快指针负责遍历
        int slow=-1,high=0;
        for(;high<nums.length;high++){
            if(high==0||nums[high-1]!=nums[high]){
                nums[++slow]=nums[high];
            }
        }
        return slow+1;
    }
~~~



#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

难度困难1443

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

**示例 1：**

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

**示例 2：**

```
输入：lists = []
输出：[]
```

**示例 3：**

```
输入：lists = [[]]
输出：[]
```

 

**提示：**

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` 按 **升序** 排列
- `lists[i].length` 的总和不超过 `10^4`

通过次数298,580

提交次数535,925



**堆排序问题**

设链表平均长度是$n$,一共有$k$个链表进行排序，按照常规的做法，将这些链表进行顺序连接，形成一个大的链表。时间复杂度是$o(nk^2)$，这种时间复杂度无法ac。想到使用堆排序方式，一次性对$k$个链表的表头进行排序，每次去这$k$个表头所在节点的最小值，然后当前链表前移一位。时间复杂度是$o(nklogk)$。

~~~java
public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>((o1, o2) -> o1.val - 			o2.val);
        for (ListNode list : lists) {
            if (list != null) {
                priorityQueue.offer(list);
            }
        }
        ListNode dummyHead=new ListNode();
        ListNode p=dummyHead;
        while(!priorityQueue.isEmpty()){
            ListNode node=priorityQueue.poll();
            p.next=node;
            p=p.next;
            if(node.next!=null){
                priorityQueue.offer(node.next);
            }
        }
        return dummyHead.next;
    }
~~~



**官方分支合并算法**

~~~java
 public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length - 1);
    }

    public ListNode merge(ListNode[] lists, int l, int r) {
        if (l == r) {
            return lists[l];
        }
        if (l > r) {
            return null;
        }
        int mid = (l + r) >> 1;
        return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    public ListNode mergeTwoLists(ListNode a, ListNode b) {
        if (a == null || b == null) {
            return a != null ? a : b;
        }
        ListNode head = new ListNode(0);
        ListNode tail = head, aPtr = a, bPtr = b;
        while (aPtr != null && bPtr != null) {
            if (aPtr.val < bPtr.val) {
                tail.next = aPtr;
                aPtr = aPtr.next;
            } else {
                tail.next = bPtr;
                bPtr = bPtr.next;
            }
            tail = tail.next;
        }
        tail.next = (aPtr != null ? aPtr : bPtr);
        return head.next;
    }
~~~



#### [172. 阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

难度简单495

给定一个整数 *n*，返回 *n*! 结果尾数中零的数量。

**示例 1:**

```
输入: 3
输出: 0
解释: 3! = 6, 尾数中没有零。
```

**示例 2:**

```
输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.
```

**说明:** 你算法的时间复杂度应为 *O*(log *n*) 。

**我的解法**

其实就是找出数字$1...n$有多少个被5整除的次数，比如25能够兑换成2个5。

~~~
  public int trailingZeroes(int n) {
       int ans=0;
       while((n=n/5)>=1){
           ans+=n;
       }
       return ans;
    }
~~~



#### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

难度简单453

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

 

**说明：**

- 你只能使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
- 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

 

**进阶：**

- 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。

 

**示例：**

```
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```



 

**提示：**

- `1 <= x <= 9`
- 最多调用 `100` 次 `push`、`pop`、`peek` 和 `empty`
- 假设所有操作都是有效的 （例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）

通过次数145,727

提交次数211,078

**我的解法**

代码里面有详细的注释，不说明了。

~~~java
class MyQueue {

    //使用两个栈结构来实现，栈1表示队尾，栈2表示队头
    //添加元素时候，从栈1进去，取的元素从栈2取，如果栈2为空，将栈1的所有元素转移到栈2
    Stack<Integer> s1=new Stack();
    Stack<Integer> s2=new Stack();
    /** Initialize your data structure here. */
    public MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        s1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(!s2.isEmpty()){
            return s2.pop();
        }
        transfer();
        return s2.pop();
    }
    public void transfer(){
        while(!s1.isEmpty()){
            s2.push(s1.pop());
        }
    }
    /** Get the front element. */
    public int peek() {
        if(!s2.isEmpty()){
            return s2.peek();
        }
        transfer();
        return s2.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return s1.isEmpty()&&s2.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
~~~



#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

难度中等1790

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**示例 2：**

```
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

 

**提示：**

- `1 <= nums.length <= 2500`
- `-104 <= nums[i] <= 104`

 

**进阶：**

- 你可以设计时间复杂度为 `O(n2)` 的解决方案吗？
- 你能将算法的时间复杂度降低到 `O(n log(n))` 吗?

通过次数321,940

提交次数632,904

**简单dp算法**

~~~java
 public int lengthOfLIS(int[] nums) {
        int ans=1;
        int[] dp= new int[nums.length];
        dp[0]=1;
        for(int i=1;i<nums.length;i++){
            dp[i]=1;
            for(int j=0;j<i;j++){
                if(nums[j]<nums[i]){
                    dp[i]=Math.max(dp[j]+1,dp[i]);
                }
            }
            ans=Math.max(ans,dp[i]);
        }
        return ans;
    }
~~~



#### [387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

难度简单425

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

 

**示例：**

```
s = "leetcode"
返回 0

s = "loveleetcode"
返回 2
```

 

**提示：**你可以假定该字符串只包含小写字母。

通过次数205,124

提交次数387,718

**我的解答**

最直接的做法是遍历整个字符串，使用一个有序map结构统计每个字符出现的次数，然后顺序遍历map找到第一个次数为1的字符，如果有的话那就返回当前索引。还有一种方式是直接用字符数组来进行统计，相对于map结构来说更简单，时间复杂度严格是$O(n)$，代码如下

~~~java
public int firstUniqChar(String s) {
        int[] flag=new int[128];
        Arrays.fill(flag,-1);
        for(int i=0;i<s.length();i++){
            int ch=s.charAt(i);
            if(flag[ch]==-1){
                //首次遇到，设置为0
                flag[ch]=0;
            }else {
                //重复不止一次
                flag[ch]=1;
            }
        }
        for(int i=0;i<s.length();i++){
            int ch=s.charAt(i);
            if(flag[ch]==0){
                return i;
            }
        }
        return -1;
    }
~~~



#### [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

难度简单1453

罗马数字包含以下七种字符: `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

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

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做 `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

 

**示例 1:**

```
输入: "III"
输出: 3
```

**示例 2:**

```
输入: "IV"
输出: 4
```

**示例 3:**

```
输入: "IX"
输出: 9
```

**示例 4:**

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```

**示例 5:**

```
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

 

**提示：**

- `1 <= s.length <= 15`
- `s` 仅含字符 `('I', 'V', 'X', 'L', 'C', 'D', 'M')`
- 题目数据保证 `s` 是一个有效的罗马数字，且表示整数在范围 `[1, 3999]` 内
- 题目所给测试用例皆符合罗马数字书写规则，不会出现跨位等情况。
- IL 和 IM 这样的例子并不符合题目要求，49 应该写作 XLIX，999 应该写作 CMXCIX 。
- 关于罗马数字的详尽书写规则，可以参考 [罗马数字 - Mathematics ](https://b2b.partcommunity.com/community/knowledge/zh_CN/detail/10753/罗马数字#knowledge_article)。

通过次数441,827

提交次数697,718

**我的解答**

常规解法，比较直接，优先匹配双字符。

~~~java
 public int romanToInt(String s) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};    
        String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        List<String> twoSymbolChar=new ArrayList();
        twoSymbolChar.add("CM");
        twoSymbolChar.add("CD");
        twoSymbolChar.add("XC");
        twoSymbolChar.add("XL");
        twoSymbolChar.add("IX");
        twoSymbolChar.add("IV");
        int i = 0,j=0;
        int ans=0;
        for(i=0;i<s.length();i++){
            String ch=s.substring(i,i+1);
            //优先匹配两字符
            if(i<s.length()-1&&twoSymbolChar.contains(s.substring(i,i+2))){
                ch=s.substring(i,i+2);
                i++;
            }
            while(j<symbols.length){
                if(ch.equals(symbols[j])){
                    ans+=values[j];
                    break;
                }
                j++;
            }
        }
        return ans;
    }
~~~



#### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

难度简单1167收藏分享切换为英文接收动态反馈

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 `true` 。 否则，返回 `false` 。

 

**进阶：**

你能用 *O(1)*（即，常量）内存解决此问题吗？

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

**示例 3：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

 

**提示：**

- 链表中节点的数目范围是 `[0, 104]`
- `-105 <= Node.val <= 105`
- `pos` 为 `-1` 或者链表中的一个 **有效索引** 。

通过次数486,986

提交次数951,365

**我的解法**

简单的快慢指针问题。

~~~java
 public boolean hasCycle(ListNode head) {
       //简单的快慢指针问题
       if(head==null||head.next==null) {
           return false;
       }
       ListNode fast=head,slow=head;
       while(fast!=null&&fast.next!=null){
           slow=slow.next;
           fast=fast.next.next;
           if(fast==slow){
               return true;
           }
       }
       return false;
    }
~~~



#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

难度困难972收藏分享切换为英文接收动态反馈

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)

```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```
输入：n = 1
输出：[["Q"]]
```

 

**提示：**

- `1 <= n <= 9`
- 皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。

通过次数142,781

提交次数193,324

**回溯方法和减枝方法的运用**

皇后位置冲突的条件一共有3中，分别是行冲突、列冲突以及斜线冲突，行列冲突比较好解决，斜线冲突又如何表示，从定义上来说，斜线冲突表示的是两个皇后所在的位置构成的直线斜率是1或者-1。假设两个皇后位置冲突，索引位置分别是$(i,j)$和$(m,n)$，那么满足如下定义：字符输入使用的是字符数组
$$
n-j=(-1)\times(m-i)等价于m+n=i+j\\
或者
n-j=m-i等价于n+i=m+j
$$

~~~java
public void nQueens(int row,List<int[]> result,int n){
        if(row==n){
            ans.add(new ArrayList(result));
            return;
        }
        for(int j=0;j<n;j++){
            boolean conflict=false;
            for(int[] loc:result){
                //竖线或者斜线冲突
                if(loc[1]==j||row+j==loc[0]+loc[1]||row+loc[1]==j+loc[0]){
                    conflict=true;
                    break;
                }
            }
            if(!conflict){
                result.add(new int[]{row,j});
                nQueens(row+1,result,n);
                result.remove(result.size()-1);
            }
        }
    }

    public List<List<String>> solveNQueens(int n) {
        nQueens(0,new ArrayList(),n);
        List<List<String>> allPath=new ArrayList();
        //替换Q
        for(List<int[]> path:ans){
            List<String> singlePath=new ArrayList();
            for(int[] rowLoc:path){
                char[] chArr=new char[n];
                Arrays.fill(chArr,'.');
                chArr[rowLoc[1]]='Q';
                singlePath.add(new String(chArr));
            }
            allPath.add(singlePath);
        }
        return allPath;
    }
~~~



**其它解法（直接愿地修改字符）**

~~~java
  List<List<String>> res = new ArrayList<>();   //记录最终结果

    public List<List<String>> solveNQueens(int n) {
        char[][] chessboard = new char[n][n];
        //初始化棋盘
        for (char[] c : chessboard) {
            Arrays.fill(c, '.');
        }
        backTrack(n, 0, chessboard);
        return res;
    }

    public void backTrack(int n, int row, char[][] chessboard) {
        if (row == n) {    //递归终止条件
             List<String> list = new ArrayList<>();
              for (char[] c : chessboard) {
                    list.add(String.copyValueOf(c));
                }
            res.add(list);
            return;
        }

        for (int col = 0;col < n; ++col) {
            //排除不合法选择
            if (isValid (row, col, n, chessboard)) {
                //做选择，放置皇后
                chessboard[row][col] = 'Q';   
                //进入下一个决策
                backTrack(n, row+1, chessboard);
                //撤销选择，撤销皇后  
                chessboard[row][col] = '.';
            }
        }

    }

    //检查是否可以在 chessboard[row][col]放置皇后
    public boolean isValid(int row, int col, int n, char[][] chessboard) {
        // 检查列
        for (int i=0; i<n; ++i) {  //剪枝
            if (chessboard[i][col] == 'Q') {
                return false;
            }
        }
        // 检查45度对角线
        for (int i=row-1, j=col-1; i>=0 && j>=0; i--, j--) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }
        // 检查135度对角线
        for (int i=row-1, j=col+1; i>=0 && j<=n-1; i--, j++) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
~~~



#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

难度中等919收藏分享切换为英文接收动态反馈

给你一个由 `n` 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且不重复的四元组 `[nums[a], nums[b], nums[c], nums[d]]` ：

- `0 <= a, b, c, d < n`
- `a`、`b`、`c` 和 `d` **互不相同**
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

你可以按 **任意顺序** 返回答案 。

 

**示例 1：**

```
输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

**示例 2：**

```
输入：nums = [2,2,2,2,2], target = 8
输出：[[2,2,2,2]]
```

 

**提示：**

- `1 <= nums.length <= 200`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`

通过次数206,634

提交次数512,530

**通用解法**

从两数之和、三数之和一直到四数之和，有一个通用的解法（两数之和算法复杂度是$o(n)$），即求解k个数的和，使得这k个数的和等于目标数。通用的解法步骤是将数组进行排序，然后依次递归求解k个数的和、k-1个数的和一直到2个数的和。

**注意数据的去重**

~~~java
 public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return getKthSum(nums,0,target,4);
    }
    public List<List<Integer>> twoSum(int[] nums,int start,int target){
        List<List<Integer>> ans=new ArrayList();
        int end=nums.length-1;
        while(start<end){
            int sum=nums[start]+nums[end];
            if(target>sum){
                start++;
            }else if(target<sum){
                end--;
            }else{
                List<Integer> item=new ArrayList();
                item.add(nums[start]);
                item.add(nums[end]);
                ans.add(item);
                start++;
                end--;
                while(start<end&&nums[start-1]==nums[start]){
                    start++;
                }
                while(start<end&&nums[end+1]==nums[end]){
                    end--;
                }
            }
        }
        return ans;
    }
    /**
      k个数的和等于target，要求nums是排序的并且k>=2
      
     */
    public List<List<Integer>> getKthSum(int[] nums,int start,int target,int k){
        if(k==2){
            return twoSum(nums,start,target);
        }
        List<List<Integer>> ans=new ArrayList();
        for(int i=start;i+k<=nums.length;i++){
            //去掉重复
            if(i>start&&nums[i]==nums[i-1]){
                continue;
            }
            //减枝
            if(nums[i]>target&&nums[i]>=0){
                continue;
            }
            List<List<Integer>> preSum=getKthSum(nums,i+1,target-nums[i],k-1);
            if(!preSum.isEmpty()){
                for(List<Integer> entry:preSum){
                    entry.add(0,nums[i]);
                    ans.add(entry);
                }
            }
        }
        return ans;
    }
~~~



**官方解答（多增加了一个减枝）**

~~~java
 public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> quadruplets = new ArrayList<List<Integer>>();
        if (nums == null || nums.length < 4) {
            return quadruplets;
        }
        Arrays.sort(nums);
        int length = nums.length;
        for (int i = 0; i < length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            //增加了这个减枝
            if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if (nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target) {
                continue;
            }
            for (int j = i + 1; j < length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                    break;
                }
                if (nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target) {
                    continue;
                }
                int left = j + 1, right = length - 1;
                while (left < right) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        quadruplets.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        left++;
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return quadruplets;
    }

~~~



#### [819. 最常见的单词](https://leetcode-cn.com/problems/most-common-word/)

难度简单101收藏分享切换为英文接收动态反馈

给定一个段落 (paragraph) 和一个禁用单词列表 (banned)。返回出现次数最多，同时不在禁用列表中的单词。

题目保证至少有一个词不在禁用列表中，而且答案唯一。

禁用列表中的单词用小写字母表示，不含标点符号。段落中的单词不区分大小写。答案都是小写字母。

 

**示例：**

```
输入: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
输出: "ball"
解释: 
"hit" 出现了3次，但它是一个禁用的单词。
"ball" 出现了2次 (同时没有其他单词出现2次)，所以它是段落里出现次数最多的，且不在禁用列表中的单词。 
注意，所有这些单词在段落里不区分大小写，标点符号需要忽略（即使是紧挨着单词也忽略， 比如 "ball,"）， 
"hit"不是最终的答案，虽然它出现次数更多，但它在禁用单词列表中。
```

 

**提示：**

- `1 <= 段落长度 <= 1000`
- `0 <= 禁用单词个数 <= 100`
- `1 <= 禁用单词长度 <= 10`
- 答案是唯一的, 且都是小写字母 (即使在 `paragraph` 里是大写的，即使是一些特定的名词，答案都是小写的。)
- `paragraph` 只包含字母、空格和下列标点符号`!?',;.`
- 不存在没有连字符或者带有连字符的单词。
- 单词里只包含字母，不会出现省略号或者其他标点符号。

通过次数20,578

提交次数48,963

**直接解法**

统计词频，并进行单词过滤。

~~~java
  public List<String> split(String paragraph){
        int slow=0, fast=0;
        List<String> wordLists=new ArrayList();
        while(fast<paragraph.length()){
            //seek the first char pos
            while(fast<paragraph.length()&&!Character.isAlphabetic(paragraph.charAt(fast))){
                fast++;
            }
            if(fast==paragraph.length()){
                break;
            }
            slow=fast;
            while(fast<paragraph.length()&&Character.isAlphabetic(paragraph.charAt(fast))){
                fast++;
            }
            wordLists.add(paragraph.substring(slow,fast));
        }
        return wordLists;
    }
    public String mostCommonWord(String paragraph, String[] banned) {
        List<String> splitWords=split(paragraph);
        //统计词频
        Map<String,AtomicInteger> mp=new HashMap();
        Set<String> bandWordSet=new HashSet(Arrays.asList(banned));
        String ans=null;
        int maxFreq=0;
        for(String word:splitWords){
            String lowerCase=word.toLowerCase();
            if(bandWordSet.contains(lowerCase)){
                continue;
            }
            int freq=mp.computeIfAbsent(lowerCase,k->new AtomicInteger()).incrementAndGet();
            if(freq>maxFreq){
                ans=lowerCase;
                maxFreq=freq;
            }
        }
        return ans;
    }
~~~



**官方版本**

~~~java
ublic String mostCommonWord(String paragraph, String[] banned) {
        Set<String> set = new HashSet<>(Arrays.asList(banned));
        Map<String, Integer> map = new HashMap<>();
        while(start < paragraph.length()) {
            String word = getNextWord(paragraph);
            //System.out.println(word);
            if(set.contains(word)) {
                continue;
            }
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        int max = 0;
        String res = null;
        for(String word : map.keySet()) {
            int count = map.get(word);
            if(count > max) {
                max = count;
                res = word;
            }
        }
        
        return res;
    }

    public String getNextWord(String str) {
        StringBuilder sb = new StringBuilder();
        for(; start < str.length(); start++) {
            if(Character.isAlphabetic(str.charAt(start))) {
                sb.append(Character.toLowerCase(str.charAt(start)));
            } else {
                while(start < str.length() && !Character.isAlphabetic(str.charAt(start))) {
                    start++;
                }
                break;
            }
        }
        return sb.toString();
    }
~~~



#### [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

难度中等699收藏分享切换为英文接收动态反馈

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

 

**示例 1：**

```
输入：nums = [2,2,3,2]
输出：3
```

**示例 2：**

```
输入：nums = [0,1,0,1,0,1,99]
输出：99
```

 

**提示：**

- `1 <= nums.length <= 3 * 104`
- `-231 <= nums[i] <= 231 - 1`
- `nums` 中，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次**

 

**进阶：**你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

通过次数96,019

提交次数133,467

**传统笨方法**

~~~java
 public int singleNumber(int[] nums) {
        //传统笨方法，使用map统计词频
        Map<Integer,AtomicInteger> mp=new HashMap();
        for(int num:nums){
            mp.computeIfAbsent(num,k->new AtomicInteger()).incrementAndGet();
        }
        for(Map.Entry<Integer,AtomicInteger> entry:mp.entrySet()){
            if(entry.getValue().get()==1){
                return entry.getKey();
            }
        }
        return -1;
    }
~~~



**位移动运算**

~~~java
public int singleNumber(int[] nums) {
       //位运算，搜集所有整数的二进制位表示，然后统计每个位的和，之后对3进行取余
       int[] bit=new int[32];
       for(int i=0;i<32;i++){
           bit[i]=0;
           for(int j=0;j<nums.length;j++){
               bit[i]+=(nums[j]>>i)&1;
           }
       }
       int ans=0;
       for(int i=0;i<32;i++){
           ans+=(bit[i]%3)<<i;
       }
       return ans;
    }
~~~

**官方版本**

由于数组中的元素都在 $int$（即$32$位整数）范围内，因此我们可以依次计算答案的每一个二进制位是 $0$ 还是 $1$。

具体地，考虑答案的第 $i$ 个二进制位（$i$ 从 $0$ 开始编号），它可能为 $0$ 或 $1$。对于数组中非答案的元素，每一个元素都出现了 $3$ 次，对应着第 $i$ 个二进制位的 $3$ 个 $0$ 或 $3$ 个 $1$，无论是哪一种情况，它们的和都是 $3$ 的倍数（即和为 $0$ 或 $3$）。因此：

答案的第 $i$ 个二进制位就是数组中所有元素的第 $i$ 个二进制位之和除以 $3$ 的余数。

这样一来，对于数组中的每一个元素 $x$，我们使用位运算 $(x >> i) \& 1$ 得到 $x$ 的第 $i$ 个二进制位，并将它们相加再对 $3$ 取余，得到的结果一定为 $0$ 或 $1$，即为答案的第 $i$ 个二进制位。



~~~java
public int singleNumber(int[] nums) {
        int ans = 0;
        for (int i = 0; i < 32; ++i) {
            int total = 0;
            for (int num: nums) {
                total += ((num >> i) & 1);
            }
            if (total % 3 != 0) {
                ans |= (1 << i);
            }
        }
        return ans;
    }
~~~



#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

难度中等1234收藏分享切换为英文接收动态反馈

给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

 

**示例 1:**

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

通过次数168,020

提交次数400,358

**传统dp算法**

dp定义两个函数，表示以索引$i$结尾的最大或者最小的子序列成积。

~~~java
 public int maxProduct(int[] nums) {
        int preMin=nums[0],preMax=nums[0];
        int ans=nums[0];
        for(int i=1;i<nums.length;i++){
            int tmp=Math.min(nums[i],Math.min(preMin*nums[i],preMax*nums[i]));
            preMax=Math.max(nums[i],Math.max(preMin*nums[i],preMax*nums[i]));
            preMin=tmp;
            ans=Math.max(preMax,ans);
        }
        return ans;
    }
~~~







#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

难度困难990收藏分享切换为英文接收动态反馈

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

**示例 2：**

```
输入：matrix = []
输出：0
```

**示例 3：**

```
输入：matrix = [["0"]]
输出：0
```

**示例 4：**

```
输入：matrix = [["1"]]
输出：1
```

**示例 5：**

```
输入：matrix = [["0","0"]]
输出：0
```

 

**提示：**

- `rows == matrix.length`
- `cols == matrix[0].length`
- `0 <= row, cols <= 200`
- `matrix[i][j]` 为 `'0'` 或 `'1'`

通过次数91,005

提交次数176,201

























#### [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

难度中等1068收藏分享切换为英文接收动态反馈

给定一个数组，将数组中的元素向右移动 `k` 个位置，其中 `k` 是非负数。

 

**进阶：**

- 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
- 你可以使用空间复杂度为 O(1) 的 **原地** 算法解决这个问题吗？

 

**示例 1:**

```
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]
```

**示例 2:**

```
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右旋转 1 步: [99,-1,-100,3]
向右旋转 2 步: [3,99,-1,-100]
```

 

**提示：**

- `1 <= nums.length <= 2 * 104`
- `-231 <= nums[i] <= 231 - 1`
- `0 <= k <= 105`



通过次数311,644

提交次数687,812

**直接做法，时间复杂读是$O(kn)$**

向右移动k位相当于，不断的将数组末尾的数添加到数组头部，整体数据向右移动，该方法时间复杂度是$O(kn)$，超时。

~~~java
 public void rotate(int[] nums, int k) {
        //时间复杂度是O(kn)，向右移动相当于除了最后一位不断添加到首位
        k=k%nums.length;
        for(int i=0;i<k;i++){
            int last=nums[nums.length-1];
            //向后移动
            for(int j=nums.length-2;j>=0;j--){
                nums[j+1]=nums[j];
            }
            nums[0]=last;
        }
    }
~~~



**使用额外的空间**

使用额外$O(k)$的空间来存放数组倒数k个元素

~~~java
 public void rotate(int[] nums, int k) {
       //使用额外的空间，用来存放数组最后k个数
       k=k%nums.length;
       int[] extraNum=new int[k];
       for(int j=nums.length-1;j>=nums.length-k;j--){
           extraNum[j+k-nums.length]=nums[j];
       }
       //将索引0到nums.length-1-k个数移动到最终的位置
       int i=nums.length-1-k,j=nums.length-1;
       while(i>=0){
           nums[j--]=nums[i--];
       }
       for(i=0;i<k;i++){
           nums[i]=extraNum[i];
       }
    }
~~~



**官方解答数组反转**
该方法基于如下的事实：当我们将数组的元素向右移动 $k$ 次后，尾部 $kmodn$ 个元素会移动至数组头部，其余元素向后移动 $kmodn$个位置。

该方法为数组的翻转：我们可以先将所有元素翻转，这样尾部的$kmodn$ 个元素就被移至数组头部，然后我们再翻转 $[0, k\bmod n-1]$ 区间的元素和 $[k\bmod n, n-1]$ 区间的元素即能得到最后的答案。

我们以 $n=7，k=3$为例进行如下展示：

操作			结果
原始数组	$1 2 3 4 5 6 7$
翻转所有元素	$7 6 5 4 3 2 1$
翻转 $[0, k\bmod n - 1]$ 区间的元素  $ 5 6 7 4 3 2 1$
翻转 $[k\bmod n, n - 1]$ 区间的元素	$5 6 7 1 2 3 4$

~~~java
public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }
~~~



#### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

难度简单988收藏分享切换为英文接收动态反馈

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

- `push(x)` —— 将元素 x 推入栈中。
- `pop()` —— 删除栈顶的元素。
- `top()` —— 获取栈顶元素。
- `getMin()` —— 检索栈中的最小元素。

 

**示例:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

 

**提示：**

- `pop`、`top` 和 `getMin` 操作总是在 **非空栈** 上调用。

通过次数270,997

提交次数474,004

**直接做法**

传统的push、pop以及top方法直接可以通过栈来实现，getMin无法通过一个栈来实现，考虑增加一个栈结构，增加一个单调栈，栈里面存放的数字是单调递减（可以存在重复的数字），入栈出栈的时候维护这个单调栈结构即可。

~~~java
class MinStack {
    /**
        涉及两个栈，一个是单调递减栈（可以保存相同的元素）s1，另一个是传统的栈s2。
        s1是用获取最小的元素，s2是传统的栈，用来支持pop、push、top操作。
     */
    Stack<Integer> minSt=new Stack();
    Stack<Integer> st=new Stack();
    /** initialize your data structure here. */
    public MinStack() {

    }
    
    public void push(int val) {
        st.push(val);
        //单调栈的栈顶元素大于等于当前的元素时候，入栈
        if(minSt.isEmpty()||minSt.peek()>=val){
            minSt.push(val);
        }
    }
    
    public void pop() {
        int val=st.pop();
        //查看该值是否是最小值，如果是最小值，出栈
        if(val==minSt.peek()){
            minSt.pop();
        }
    }
    
    public int top() {
        return st.peek();
    }
    
    public int getMin() {
        return minSt.peek();
    }
}
~~~



**官方版本**

~~~java
class MinStack {

    /** initialize your data structure here. */
    Deque<Integer> xStack;
    Deque<Integer> minStack;

    public MinStack(){
        xStack = new LinkedList<Integer>();
        minStack = new LinkedList<Integer>();
        minStack.push(Integer.MAX_VALUE);
    }
    public void push(int val){
        xStack.push(val);
        minStack.push(Math.min(minStack.peek(),val));
    }
    public void pop(){
        xStack.pop();
        minStack.pop();
    }
    public int top(){
        return xStack.peek();
    }
    public int getMin(){
        return minStack.peek();
    }
}
~~~

#### [268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)

难度简单436

给定一个包含 `[0, n]` 中 `n` 个数的数组 `nums` ，找出 `[0, n]` 这个范围内没有出现在数组中的那个数。

 

**进阶：**

- 你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?

 

**示例 1：**

```
输入：nums = [3,0,1]
输出：2
解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 2：**

```
输入：nums = [0,1]
输出：2
解释：n = 2，因为有 2 个数字，所以所有的数字都在范围 [0,2] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 3：**

```
输入：nums = [9,6,4,2,3,5,7,0,1]
输出：8
解释：n = 9，因为有 9 个数字，所以所有的数字都在范围 [0,9] 内。8 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 4：**

```
输入：nums = [0]
输出：1
解释：n = 1，因为有 1 个数字，所以所有的数字都在范围 [0,1] 内。1 是丢失的数字，因为它没有出现在 nums 中。
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 104`
- `0 <= nums[i] <= n`
- `nums` 中的所有数字都 **独一无二**

通过次数143,637

提交次数230,966

**方法一数组交换**

交换数组元素到指定的位置，对于数组中的元素$k$将其替换到索引$k$处。

~~~java
 public int missingNumber(int[] nums) {
        int n=nums.length;
        //交换数组内容，将位置i的值设置成i
        for(int i=0;i<n;i++){
            while(nums[i]>=0&&nums[i]<n&&nums[i]!=i){
                int tmp=nums[i];
                nums[i]=nums[tmp];
                nums[tmp]=tmp;
            }
        }
        for(int i=0;i<n;i++){
            if(nums[i]!=i){
                return i;
            }
        }
        return n;
    }
~~~



**方法二求和**

题目描述只缺少一个数，数字范围是$0...n+1$之间（$n$是数组长度），那么数字$0...n+1$的和减去当前数组的元素和就是缺失的那个数。

~~~java
 public int missingNumber(int[] nums) {
        int sum = 0;
        for (int i=0; i < nums.length; i++){
            sum += nums[i];
        }
        int maxsum = (nums.length)*(nums.length+1)/2;
        return maxsum-sum;
    }
~~~



**方法三位运算**

~~~java
 public int missingNumber(int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; i++) {
            missing ^= i ^ nums[i];
        }
        return missing;
    }
~~~



#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

难度中等1353

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `1` 到 `n` 之间（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，找出 **这个重复的数** 。

你设计的解决方案必须不修改数组 `nums` 且只用常量级 `O(1)` 的额外空间。

 

**示例 1：**

```
输入：nums = [1,3,4,2,2]
输出：2
```

**示例 2：**

```
输入：nums = [3,1,3,4,2]
输出：3
```

**示例 3：**

```
输入：nums = [1,1]
输出：1
```

**示例 4：**

```
输入：nums = [1,1,2]
输出：1
```

 

**提示：**

- `1 <= n <= 105`
- `nums.length == n + 1`
- `1 <= nums[i] <= n`
- `nums` 中 **只有一个整数** 出现 **两次或多次** ，其余整数均只出现 **一次**

 

**进阶：**

- 如何证明 `nums` 中至少存在一个重复的数字?
- 你可以设计一个线性级时间复杂度 `O(n)` 的解决方案吗？

通过次数169,269

提交次数255,175

**方法一交换数组**

思路和前面一提相似。

~~~java
 public int findDuplicate(int[] nums) {
        for(int i=0;i<nums.length;i++){
            int tmp=nums[i];
            while(tmp!=i+1){
                if(nums[tmp-1]==tmp){
                    return tmp;
                }
                nums[i]=nums[tmp-1];
                nums[tmp-1]=tmp;
                tmp=nums[i];
            }
        }
        return -1;
    }
~~~

#### [318. 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)

难度中等182

给定一个字符串数组 `words`，找到 `length(word[i]) * length(word[j])` 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。

 

**示例 1:**

```
输入: ["abcw","baz","foo","bar","xtfn","abcdef"]
输出: 16 
解释: 这两个单词为 "abcw", "xtfn"。
```

**示例 2:**

```
输入: ["a","ab","abc","d","cd","bcd","abcd"]
输出: 4 
解释: 这两个单词为 "ab", "cd"。
```

**示例 3:**

```
输入: ["a","aa","aaa","aaaa"]
输出: 0 
解释: 不存在这样的两个单词。
```

 

**提示：**

- `2 <= words.length <= 1000`
- `1 <= words[i].length <= 1000`
- `words[i]` 仅包含小写字母

通过次数18,569

提交次数27,254

**常规解法**

主要是改进比较两个单词是否包含公共单词方法。

~~~java
 /**
    ** 检测两个单词是否包含相同的字母
     */
    boolean containsCommonLetter(String word1,String word2){
        int[] freq=new int[26];
        for(int i=0;i<word1.length();i++){
            freq[word1.charAt(i)-'a']+=1;
        }
        for(int i=0;i<word2.length();i++){
            if(freq[word2.charAt(i)-'a']>0){
                return true;
            }
        }
        return false;
    }
    public int maxProduct(String[] words) {
        int ans=0;
        for(int i=0;i<words.length;i++){
            for(int j=i+1;j<words.length;j++){
                //减枝
                if(words[i].length()<=ans/words[j].length()){
                    continue;
                }
                if(containsCommonLetter(words[i],words[j])){
                    continue;
                }
                ans=Math.max(ans,words[i].length()*words[j].length());
            }
        }
        return ans;
    }
~~~



**位操作+预计算+hashmap**

~~~java
  public int bitNumber(char ch){
    return (int)ch - (int)'a';
  }

  public int maxProduct(String[] words) {
    Map<Integer, Integer> hashmap = new HashMap();

    int bitmask = 0, bitNum = 0;
    for (String word : words) {
      bitmask = 0;
      for (char ch : word.toCharArray()) {
        // add bit number bitNumber in bitmask
        bitmask |= 1 << bitNumber(ch);
      }
      // there could be different words with the same bitmask
      // ex. ab and aabb
      hashmap.put(bitmask, Math.max(hashmap.getOrDefault(bitmask, 0), word.length()));
    }

    int maxProd = 0;
    for (int x : hashmap.keySet())
      for (int y : hashmap.keySet())
        if ((x & y) == 0) maxProd = Math.max(maxProd, hashmap.get(x) * hashmap.get(y));

    return maxProd;
  }
~~~



#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

难度中等775

给定一个可包含重复数字的序列 `nums` ，**按任意顺序** 返回所有不重复的全排列。

 

**示例 1：**

```
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

**示例 2：**

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

 

**提示：**

- `1 <= nums.length <= 8`
- `-10 <= nums[i] <= 10`

通过次数198,803

提交次数312,765

**我的解法**

和全排列I的思路类似，采用回溯方法，每次迭代过程中增加一个数字，增加数字完成之后为了避免再次碰到该数字，进行数字交换。因为输入包含重复数字，需要进行去重标记，由于数字范围是$[-10,10]$，直接使用一个byte数组进行标记。

~~~java
public List<List<Integer>> ans=new ArrayList();
    public void permute(List<Integer> numList,int start,List<Integer> result){
        if(start==numList.size()){
            ans.add(new ArrayList(result));
            return;
        }
        //用来标记当前函数执行过程中已经选择过的第start个数字
        byte[] bit=new byte[21];
        for(int i=start;i<numList.size();i++){
            if(bit[numList.get(i)+10]==1){
                continue;
            }
            //标记当前数字已经选择，数字区间范围是[-10,10]
            bit[numList.get(i)+10]=1;
            result.add(numList.get(i));
            Collections.swap(numList,i,start);
            permute(numList,start+1,result);
            Collections.swap(numList,i,start);
            result.remove(result.size()-1);
        }
    }
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<Integer> numList=Arrays.stream(nums).boxed().collect(Collectors.toList());
        permute(numList,0,new ArrayList());
        return ans;
    }
~~~



**官方答案：标记回溯**

~~~java
  boolean[] vis;

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> perm = new ArrayList<Integer>();
        vis = new boolean[nums.length];
        Arrays.sort(nums);
        backtrack(nums, ans, 0, perm);
        return ans;
    }

    public void backtrack(int[] nums, List<List<Integer>> ans, int idx, List<Integer> perm) {
        if (idx == nums.length) {
            ans.add(new ArrayList<Integer>(perm));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                continue;
            }
            perm.add(nums[i]);
            vis[i] = true;
            backtrack(nums, ans, idx + 1, perm);
            vis[i] = false;
            perm.remove(idx);
        }
    }
~~~



#### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

难度简单997

实现 [strStr()](https://baike.baidu.com/item/strstr/811469) 函数。

给你两个字符串 `haystack` 和 `needle` ，请你在 `haystack` 字符串中找出 `needle` 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回 `-1` 。

 

**说明：**

当 `needle` 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 `needle` 是空字符串时我们应当返回 0 。这与 C 语言的 [strstr()](https://baike.baidu.com/item/strstr/811469) 以及 Java 的 [indexOf()](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html#indexOf(java.lang.String)) 定义相符。

 

**示例 1：**

```
输入：haystack = "hello", needle = "ll"
输出：2
```

**示例 2：**

```
输入：haystack = "aaaaa", needle = "bba"
输出：-1
```

**示例 3：**

```
输入：haystack = "", needle = ""
输出：0
```

 

**提示：**

- `0 <= haystack.length, needle.length <= 5 * 104`
- `haystack` 和 `needle` 仅由小写英文字符组成

通过次数443,466

提交次数1,088,995

**我的解法**

利用前缀和思想来进行求解，为了减少字符串全部比较操作，采用计算hash值的方式。

~~~java
	/**
        计算字符串hashcode值
     */
    public int[] hashcodeInteger(String str){
        int[] code=new int[str.length()+1];
        code[0]=0;
        int i=1;
        for(char ch:str.toCharArray()){
            code[i]=code[i-1]+ch*31;
            i++;
        }
        return code;
    }
    public int strStr(String haystack, String needle) {
        if(needle.length()==0){
            return 0;
        }
        int[] hayCode=hashcodeInteger(haystack);
        int needleCode=hashcodeInteger(needle)[needle.length()];
        for(int i=0;i<haystack.length()-needle.length()+1;i++){
             int subCode=hayCode[i+needle.length()]-hayCode[i];
             if(subCode!=needleCode){
                 continue;
             }
             if(haystack.substring(i,i+needle.length()).equals(needle)){
                 return i;
             }
        }
        return -1;
    }
~~~



**官方解答kmp算法**

~~~java
 public int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        if (m == 0) {
            return 0;
        }
        int[] pi = new int[m];
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && needle.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (needle.charAt(i) == needle.charAt(j)) {
                j++;
            }
            pi[i] = j;
        }
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)) {
                j++;
            }
            if (j == m) {
                return i - m + 1;
            }
        }
        return -1;
    }
~~~



#### [954. 二倍数对数组](https://leetcode-cn.com/problems/array-of-doubled-pairs/)

难度中等51

给定一个长度为偶数的整数数组 `arr`，只有对 `arr` 进行重组后可以满足 “对于每个 `0 <= i < len(arr) / 2`，都有 `arr[2 * i + 1] = 2 * arr[2 * i]`” 时，返回 `true`；否则，返回 `false`。

 

**示例 1：**

```
输入：arr = [3,1,3,6]
输出：false
```

**示例 2：**

```
输入：arr = [2,1,2,6]
输出：false
```

**示例 3：**

```
输入：arr = [4,-2,2,-4]
输出：true
解释：可以用 [-2,-4] 和 [2,4] 这两组组成 [-2,-4,2,4] 或是 [2,4,-2,-4]
```

**示例 4：**

```
输入：arr = [1,2,4,16,8,4]
输出：false
```

 

**提示：**

- `0 <= arr.length <= 3 * 104`
- `arr.length` 是偶数
- `-105 <= arr[i] <= 105`

通过次数6,794

提交次数22,753



**官方答案**

基本思路是相同的，就是按照数据的绝对值进行排序，从第一个数字进行遍历，如果当前数字$x$没有被遍历过，那么对应的$2x$必须要存在，不然它就不是对数组，遍历完之后要进行标记。

~~~java
public boolean canReorderDoubled(int[] A) {
        // count[x] = the number of occurrences of x in A
        Map<Integer, Integer> count = new HashMap();
        for (int x: A)
            count.put(x, count.getOrDefault(x, 0) + 1);

        // B = A as Integer[], sorted by absolute value
        Integer[] B = new Integer[A.length];
        for (int i = 0; i < A.length; ++i)
            B[i] = A[i];
       Arrays.sort(B, Comparator.comparingInt(Math::abs));

        for (int x: B) {
            // If this can't be consumed, skip
            if (count.get(x) == 0) continue;
            // If this doesn't have a doubled partner, the answer is false
            if (count.getOrDefault(2*x, 0) <= 0) return false;

            // Write x, 2*x
            count.put(x, count.get(x) - 1);
            count.put(2*x, count.get(2*x) - 1);
        }

        // If we have written everything, the answer is true
        return true;
    }
~~~



**我的答案双指针**

通过双指针方式进行遍历而不是采用map形式。

~~~java
public boolean canReorderDoubled(int[] arr) {
        List<Integer> arrList=Arrays.stream(arr).boxed().collect(Collectors.toList());
        //按照绝对值排序
        Collections.sort(arrList,(n1,n2)->{
            if(Math.abs(n1)!=Math.abs(n2)){
                return Math.abs(n1)-Math.abs(n2);
            }
            return n1-n2;
        });
        //采用双指针进行匹配
        //标记是否做了匹配
        int left=0;
        boolean[] matched=new boolean[arrList.size()];
        for(int i=left+1;i<arrList.size();i++){
            while(matched[left]&&left<arrList.size()){
                left++;
            }
            if(i!=left&&arrList.get(i)==2*arrList.get(left)){
                matched[left]=true;
                matched[i]=true;
                continue;
            }
        }
        for(boolean result:matched){
            if(!result){
                return false;
            }
        }
        return true;
    }
~~~



**方法三桶排序思想**

~~~java
public boolean canReorderDoubled(int[] A) {
       //桶排序，将数组分成两种类型，负数和非负数
       int extra=100000;
       int bucketSize=2*extra+1;
       int[] bucket=new int[bucketSize];
       for(int num:A){
           bucket[num+extra]++;
       } 
       //首先是0,0的个数必须是偶数
       if((bucket[extra]&2)==1){
           return false;
       }
       //对应的负数，从大到小
       for(int i=extra-1;i>=0;i--){
           if(bucket[i]==0){
               continue;
           }
           //对应的二倍数越界或者值小于，注意不是等于，例如-2,-4,-4,-8这种
           if(2*i-extra<0||bucket[2*i-extra]<bucket[i]){
               return false;
           }
           //标记
           bucket[2*i-extra]-=bucket[i];
           bucket[i]=0;
       }
       //对应的整数，从小到大
       for(int i=extra+1;i<bucketSize;i++){
           if(bucket[i]==0){
               continue;
           }
           if(2*i-extra>=bucketSize||bucket[2*i-extra]<bucket[i]){
               return false;
           }
           bucket[2*i-extra]-=bucket[i];
           bucket[i]=0;
       }
       return true;
    }
~~~



#### [66. 加一](https://leetcode-cn.com/problems/plus-one/)

难度简单733

给定一个由 **整数** 组成的 **非空** 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储**单个**数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

 

**示例 1：**

```
输入：digits = [1,2,3]
输出：[1,2,4]
解释：输入数组表示数字 123。
```

**示例 2：**

```
输入：digits = [4,3,2,1]
输出：[4,3,2,2]
解释：输入数组表示数字 4321。
```

**示例 3：**

```
输入：digits = [0]
输出：[1]
```

 

**提示：**

- `1 <= digits.length <= 100`
- `0 <= digits[i] <= 9`

通过次数332,141

提交次数724,287

**直接解法**

注意溢出位处理以及数据类型转换，int[]数组转换成List\<Integer\>。

~~~java
 public int[] plusOne(int[] digits) {
        int carry=1;
        for(int i=digits.length-1;i>=0;i--){
            int sum=carry+digits[i];
            if(sum>=10){
                carry=1;
            }else{
                carry=0;
            }
            digits[i]=sum%10;
        }
        if(carry==0){
            return digits;
        }else{
            List<Integer> digitList=Arrays.stream(digits).boxed().collect(Collectors.toList());
            digitList.add(0,1);
            return digitList.stream().mapToInt(Integer::intValue).toArray();
        }
    }
~~~



#### [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

难度中等852

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg)

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

通过次数178,421

提交次数373,162

**我的解法**

这种问题关键在于如何确定开始和结束以及循环什么时候终止。看题解

~~~java
 public List<Integer> spiralOrder(int[][] matrix) {
        //整个遍历过程是从坐标(i,i)开始，然后开始按照从左到右，
        //从上到下、从右到左、从下到上的顺序进行的。整个遍历的坐标变换依次如下
        //从左到右 (i,i)->(i,n-1-i)
        //从上到下 (i+1,n-1-i)->(m-1-i,n-1-i)
        //从右到左 (m-1-i,n-2-i)->(m-1-i,i)
        //从下到上 (m-2-i,i)->(i+1,i)
        //注意每轮循环判断当前元素是否已经遍历完
        int m=matrix.length,n=matrix[0].length;
        int total=m*n;
        List<Integer> ans=new ArrayList(total);
        int i=0;
        while(true){
            //从左到右
            for(int j=i;j<=n-1-i;j++){
                ans.add(matrix[i][j]);
            }
            if(ans.size()==total){
                break;
            }
            //从上到下
            for(int j=i+1;j<=m-1-i;j++){
                ans.add(matrix[j][n-1-i]);
            }
            if(ans.size()==total){
                break;
            }
            //从右到左
            for(int j=n-2-i;j>=i;j--){
                ans.add(matrix[m-1-i][j]);
            }
            if(ans.size()==total){
                break;
            }
            //从下到上
            for(int j=m-2-i;j>=i+1;j--){
                ans.add(matrix[j][i]);
            }
            if(ans.size()==total){
                break;
            }
            i++;
        }
        return ans;
    }
~~~



**官方解答**

通过数组标记形式。

~~~java
public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> order = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return order;
        }
        int rows = matrix.length, columns = matrix[0].length;
        boolean[][] visited = new boolean[rows][columns];
        int total = rows * columns;
        int row = 0, column = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            order.add(matrix[row][column]);
            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }
~~~



#### [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

难度困难4401

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

 

**示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

**示例 2：**

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

**示例 3：**

```
输入：nums1 = [0,0], nums2 = [0,0]
输出：0.00000
```

**示例 4：**

```
输入：nums1 = [], nums2 = [1]
输出：1.00000
```

**示例 5：**

```
输入：nums1 = [2], nums2 = []
输出：2.00000
```

 

**提示：**

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-106 <= nums1[i], nums2[i] <= 106`

 

**进阶：**你能设计一个时间复杂度为 `O(log (m+n))` 的算法解决此问题吗？

通过次数482,508

提交次数1,186,049

**折半查找**

该题比较典型，具体表现从2个有序的数组中查找第k小的数字，为了简化逻辑，索引k从1开始。注意折半查找中间数，当k为基数的时候，即使两个中间数相等，也不能直接返回结果。

~~~java
/**
     * k从索引1开始
     */
    public int findKMin(int[] nums1, int i, int j, int[] nums2, int m, int n, int k) {
        if (i > j) {
            return nums2[m + k - 1];
        }
        if (m > n) {
            return nums1[i + k - 1];
        }
        if (k == 1) {
            return Math.min(nums1[i], nums2[m]);
        }
        int n1Right = j - i >= k / 2 - 1 ? i + k / 2 - 1 : j;
        int n2Right = n - m >= k / 2 - 1 ? m + k / 2 - 1 : n;
        //折半查找
        if (nums1[n1Right] == nums2[n2Right]) {
            //k如果是奇数，那么i到n1Right和j到n2Right都要舍弃
            if (n1Right == j||(k%2==1)) {
                return findKMin(nums1, n1Right + 1, j, nums2, m, n, k - (n1Right - i + 1));
            } else if (n2Right == n) {
                return findKMin(nums1, i, j, nums2, n2Right + 1, n, k - (n2Right - m + 1));
            }
            return nums1[n1Right];
        } else if (nums1[n1Right] < nums2[n2Right]) {
            //丢弃掉i~n1Right之间的元素
            return findKMin(nums1, n1Right + 1, j, nums2, m, n, k - (n1Right - i + 1));
        } else {
            return findKMin(nums1, i, j, nums2, n2Right + 1, n, k - (n2Right - m + 1));
        }
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        int total = n1 + n2;
        int ans = 0;
        ans += findKMin(nums1, 0, n1 - 1, nums2, 0, n2 - 1, total / 2 + 1);
        if ((total % 2) == 0) {
            ans += findKMin(nums1, 0, n1 - 1, nums2, 0, n2 - 1, total / 2);
            return 1.0 * ans / 2;
        }
        return ans;
    }
~~~



**网上优秀解答**

~~~java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }
    //i: nums1的起始位置 j: nums2的起始位置
    public int findKth(int[] nums1, int i, int[] nums2, int j, int k){
        if( i >= nums1.length) return nums2[j + k - 1];//nums1为空数组
        if( j >= nums2.length) return nums1[i + k - 1];//nums2为空数组
        if(k == 1){
            return Math.min(nums1[i], nums2[j]);
        }
        int midVal1 = (i + k / 2 - 1 < nums1.length) ? nums1[i + k / 2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k / 2 - 1 < nums2.length) ? nums2[j + k / 2 - 1] : Integer.MAX_VALUE;
        if(midVal1 < midVal2){
            return findKth(nums1, i + k / 2, nums2, j , k - k / 2);
        }else{
            return findKth(nums1, i, nums2, j + k / 2 , k - k / 2);
        }        
    }
~~~



#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

难度简单1087

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

```
输入：head = [1,2,2,1]
输出：true
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg)

```
输入：head = [1,2]
输出：false
```

 

**提示：**

- 链表中节点数目在范围`[1, 105]` 内
- `0 <= Node.val <= 9`

 

**进阶：**你能否用 `O(n)` 时间复杂度和 `O(1)` 空间复杂度解决此题？

通过次数292,701

提交次数595,055

**我的解法**

回文链表简单来说，就是这个链表的正序和反序内容相同，进一步就是该链表是对称的，题目要求空间复杂度是常量级别，那么不能使用额外的空间。想到从这个链表的中间节点将该链表截断，对左半部分链表进行反转。如下所示：注意题目中如何求解中间所在节点（即第n/2个节点）

~~~xml
1->4->5->4-1
1->4	4->1	1-4左半部分进行反转得到4->1，然后和右半部分进行对比
~~~

~~~java
public ListNode reverseNode(ListNode head){
        if(head==null||head.next==null){
            return head;
        }
        ListNode next=head.next;
        ListNode reverseHead=reverseNode(next);
        next.next=head;
        head.next=null;
        return reverseHead;
    }
    public boolean isPalindrome(ListNode head) {
        if(head==null||head.next==null){
            return true;
        }
        //fast节点先前进2步
        ListNode slow=head,fast=head.next;
        //找到中间的节点，中间被slow节点指向
        while(fast.next!=null&&fast.next.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        //当前是奇数还是偶数个节点
        boolean odd=(fast.next!=null); 
        ListNode nextNode=odd?slow.next.next:slow.next;
        slow.next=null;
        //反转head到slow节点
        ListNode reverseHead=reverseNode(head);
        ListNode p=nextNode,q=reverseHead;
        while(p!=null&&q!=null){
            if(p.val!=q.val){
                return false;
            }
            p=p.next;
            q=q.next;
        }
        return p==null&&q==null;
    }
~~~

**递归解法**

~~~java
  private ListNode frontPointer;

    private boolean recursivelyCheck(ListNode currentNode) {
        if (currentNode != null) {
            if (!recursivelyCheck(currentNode.next)) {
                return false;
            }
            if (currentNode.val != frontPointer.val) {
                return false;
            }
            frontPointer = frontPointer.next;
        }
        return true;
    }

    public boolean isPalindrome(ListNode head) {
        frontPointer = head;
        return recursivelyCheck(head);
    }
~~~



#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

难度中等892

给你一个长度为 *n* 的整数数组 `nums`，其中 *n* > 1，返回输出数组 `output` ，其中 `output[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

 

**示例:**

```
输入: [1,2,3,4]
输出: [24,12,8,6]
```

 

**提示：**题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

**说明:** 请**不要使用除法，**且在 O(*n*) 时间复杂度内完成此题。

**进阶：**
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组**不被视为**额外空间。）

通过次数131,417

提交次数181,901

**传统的直接做法**

注意数字0的影响，当数组中超过1个0的时候，返回结果数组全部是0；如果只有1个0时，除了0对应的返回结果，其它都是0。

~~~java
 public int[] productExceptSelf(int[] nums) {
        AtomicInteger zeroCount=new AtomicInteger();
        int productAll=Arrays.stream(nums).boxed().collect(Collectors.toList()).stream().reduce(1,(x,y)->{
            if(y==0){
                zeroCount.incrementAndGet();
                return x;
            }
            return x*y;
        });
        int[] ans=new int[nums.length];
        if(zeroCount.get()>=2){
            return ans;
        }
        int i=0;
        for(int num:nums){
            ans[i++]=num==0?(int)productAll:zeroCount.get()>0?0:productAll/num;
        }
        return ans;
    }
~~~



**官方解答1：左右乘积列表**

~~~java
public int[] productExceptSelf(int[] nums) {
        int length = nums.length;

        // L 和 R 分别表示左右两侧的乘积列表
        int[] L = new int[length];
        int[] R = new int[length];

        int[] answer = new int[length];

        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {
            L[i] = nums[i - 1] * L[i - 1];
        }

        // R[i] 为索引 i 右侧所有元素的乘积
        // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            R[i] = nums[i + 1] * R[i + 1];
        }

        // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < length; i++) {
            answer[i] = L[i] * R[i];
        }

        return answer;
    }
~~~



**空间复杂度$o(n)$优化**

~~~java
  public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];

        // answer[i] 表示索引 i 左侧所有元素的乘积
        // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
        answer[0] = 1;
        for (int i = 1; i < length; i++) {
            answer[i] = nums[i - 1] * answer[i - 1];
        }

        // R 为右侧所有元素的乘积
        // 刚开始右边没有元素，所以 R = 1
        int R = 1;
        for (int i = length - 1; i >= 0; i--) {
            // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
            answer[i] = answer[i] * R;
            // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
            R *= nums[i];
        }
        return answer;
    }
~~~



## 2021年9月

#### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

难度简单1051

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

 

**示例 1:**

```
输入: nums = [1,3,5,6], target = 5
输出: 2
```

**示例 2:**

```
输入: nums = [1,3,5,6], target = 2
输出: 1
```

**示例 3:**

```
输入: nums = [1,3,5,6], target = 7
输出: 4
```

**示例 4:**

```
输入: nums = [1,3,5,6], target = 0
输出: 0
```

**示例 5:**

```
输入: nums = [1], target = 0
输出: 0
```

 

**提示:**

- `1 <= nums.length <= 104`
- `-104 <= nums[i] <= 104`
- `nums` 为**无重复元素**的**升序**排列数组
- `-104 <= target <= 104`

通过次数480,935

提交次数1,032,609

**直接解法**

题目比较简单，使用的是二分查找，二分查找的终止条件是$low=high+1$，从判断条件上来说如果当前的数字不在当前数组中，那么数组插入的位置一定是$low$。

~~~java
  public int searchInsert(int[] nums, int target) {
        int low=0,high=nums.length-1;
        while(low<=high){
            int mid=low+(high-low)/2;
            if(nums[mid]==target){
                return mid;
            }else if(nums[mid]>target){
                high=mid-1;
            }else{
                low=mid+1;
            }
        }
        return high+1;
    }
~~~



#### [38. 外观数列](https://leetcode-cn.com/problems/count-and-say/)

难度中等737

给定一个正整数 `n` ，输出外观数列的第 `n` 项。

「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

你可以将其视作是由递归公式定义的数字字符串序列：

- `countAndSay(1) = "1"`
- `countAndSay(n)` 是对 `countAndSay(n-1)` 的描述，然后转换成另一个数字字符串。

前五项如下：

```
1.     1
2.     11
3.     21
4.     1211
5.     111221
第一项是数字 1 
描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
```

要 **描述** 一个数字字符串，首先要将字符串分割为 **最小** 数量的组，每个组都由连续的最多 **相同字符** 组成。然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。

例如，数字字符串 `"3322251"` 的描述如下图：

![img](https://pic.leetcode-cn.com/1629874763-TGmKUh-image.png)



 

**示例 1：**

```
输入：n = 1
输出："1"
解释：这是一个基本样例。
```

**示例 2：**

```
输入：n = 4
输出："1211"
解释：
countAndSay(1) = "1"
countAndSay(2) = 读 "1" = 一 个 1 = "11"
countAndSay(3) = 读 "11" = 二 个 1 = "21"
countAndSay(4) = 读 "21" = 一 个 2 + 一 个 1 = "12" + "11" = "1211"
```

 

**提示：**

- `1 <= n <= 30`

通过次数211,339

提交次数364,490

**直接解法**

实质上是统计字符串中连续相同的数字的个数，拼接字符串的时候使用StringBuilder，提高效率，将字符串转换成char数组，从定义上来看本质就是一个递归调用。

~~~java
 public String countAndSay(int n) {
        if(n==1){
            return "1";
        }
        String preAns=countAndSay(n-1);
        StringBuilder ans=new StringBuilder();
        int count=1;
        char[] chArr=preAns.toCharArray();
        char ch=chArr[0];
        for(int i=1;i<chArr.length;i++){
            if(chArr[i]==ch){
                count++;
            }else{
                ans.append(count).append(ch-'0');
                count=1;
                ch=chArr[i];
            }
        }
        ans.append(count).append(ch-'0');
        return ans.toString();
    }
~~~

#### [278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

难度简单399

你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。

假设你有 `n` 个版本 `[1, 2, ..., n]`，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 `bool isBadVersion(version)` 接口来判断版本号 `version` 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

**示例 1：**

```
输入：n = 5, bad = 4
输出：4
解释：
调用 isBadVersion(3) -> false 
调用 isBadVersion(5) -> true 
调用 isBadVersion(4) -> true
所以，4 是第一个错误的版本。
```

**示例 2：**

```
输入：n = 1, bad = 1
输出：1
```

 

**提示：**

- `1 <= bad <= n <= 231 - 1`

通过次数166,092

提交次数365,001

**直接解法**

题目含义是找出数组中第一个最小的坏的版本，这个版本从定义上来说就是一个有序数组，所以可以采用二分查找形式。注意二分查找mid的构造中算术运算符优先级顺序，位移运算符$>>$优先级低于算术元素符$+$。

~~~java
 public int firstBadVersion(int n) {
        int badVersion=-1;
        int low=1,high=n;
        while(low<=high){
            int mid=low+((high-low)>>1);
            if(isBadVersion(mid)){
                badVersion=mid;
                high=mid-1;
            }else{
                low=mid+1;
            }
        }
        return badVersion;
    }
~~~



#### [324. 摆动排序 II](https://leetcode-cn.com/problems/wiggle-sort-ii/)

难度中等276

给你一个整数数组 `nums`，将它重新排列成 `nums[0] < nums[1] > nums[2] < nums[3]...` 的顺序。

你可以假设所有输入数组都可以得到满足题目要求的结果。

 

**示例 1：**

```
输入：nums = [1,5,1,1,6,4]
输出：[1,6,1,5,1,4]
解释：[1,4,1,5,1,6] 同样是符合题目要求的结果，可以被判题程序接受。
```

**示例 2：**

```
输入：nums = [1,3,2,2,3,1]
输出：[2,3,1,3,1,2]
```

 

**提示：**

- `1 <= nums.length <= 5 * 104`
- `0 <= nums[i] <= 5000`
- 题目数据保证，对于给定的输入 `nums` ，总能产生满足题目要求的结果

 

**进阶：**你能用 O(n) 时间复杂度和 / 或原地 O(1) 额外空间来实现吗？

通过次数24,573

提交次数65,013

**我的解答**

排序算法以及使用了额外的空间复杂度。题目要求是摆动排序，从定义上来说，就是数组按照小大小大的顺序排列。自然的一个想法是想到排序，比如数组$1,1,2,2,3,3$，排好序之后将数组分为低位和高位，低位是$1,1,2$，高位是$2,3,3$，然后依次从低位高位分别抽取数字填充到新的数组。但是对于数组$4,5,5,6$，按照这种抽取方式得到的结果是$4,5,5,6$,不满足定义。我们从右边向左边进行抽取就可以了。

~~~java
 public void wiggleSort(int[] nums) {
        Arrays.sort(nums);
        int len=nums.length;
        int i=(len+1)/2-1,j=len-1,k=0;
        int[] ans=new int[len];
        for(;i>=0;i--){
            ans[k++]=nums[i];
            if(j>(len+1)/2-1){
                ans[k++]=nums[j--];
            }
        }
        for(i=0;i<len;i++){
            nums[i]=ans[i];
        }
    }
~~~



**官网解答：快速排序**

时间复杂度是$o(n)$，基本思路是找到中间的一个元素，然后将数组划分成两部分。还有一种更优的解法，空间复杂度是常量级，采用虚拟地址映射思想，不太容易理解，不考虑了。

~~~c++
void wiggleSort(vector<int>& nums) {
        auto midptr = nums.begin() + nums.size() / 2;
        nth_element(nums.begin(), midptr, nums.end());
        int mid = *midptr;
        
        // 3-way-partition
        int i = 0, j = 0, k = nums.size() - 1;
        while(j < k){
            if(nums[j] > mid){
                swap(nums[j], nums[k]);
                --k;
            }
            else if(nums[j] < mid){
                swap(nums[j], nums[i]);
                ++i;
                ++j;
            }
            else{
                ++j;
            }
        }
        
        if(nums.size() % 2) ++midptr;
        vector<int> tmp1(nums.begin(), midptr);
        vector<int> tmp2(midptr, nums.end());
        for(int i = 0; i < tmp1.size(); ++i){
            nums[2 * i] = tmp1[tmp1.size() - 1 - i];
        }
        for(int i = 0; i < tmp2.size(); ++i){
            nums[2 * i + 1] = tmp2[tmp2.size() - 1 - i];
        }
    }
~~~





## 2022年7月

#### [389. 找不同](https://leetcode.cn/problems/find-the-difference/)

难度简单328收藏分享切换为英文接收动态反馈

给定两个字符串 `s` 和 `t` ，它们只包含小写字母。

字符串 `t` 由字符串 `s` 随机重排，然后在随机位置添加一个字母。

请找出在 `t` 中被添加的字母。

 

**示例 1：**

```
输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。
```

**示例 2：**

```
输入：s = "", t = "y"
输出："y"
```

 

**提示：**

- `0 <= s.length <= 1000`
- `t.length == s.length + 1`
- `s` 和 `t` 只包含小写字母

通过次数130,912

提交次数192,186

##### 我的解法

题目相对比较简单，按照题目大意字符串t中的字符（不考虑顺序）只是比s多了一个，这个字符可能是s中的，也可能是不在s中。简单的做法就是直接排序、统计，但是这种方法算法复杂度过高$nlgn$，可以直接进行字符统计

```java
  public char findTheDifference(String s, String t) {
        int[] chCount=new int[26];
        char[] sArr=s.toCharArray();
        char[] tArr=t.toCharArray();
        for(char ch:sArr){
            chCount[ch-'a']++;
        }
        for(char ch:tArr){
           if(chCount[ch-'a']==0){
               return ch;
           }
           chCount[ch-'a']--;
        }
        return '0';
    }
```



##### 官方解法二：求和

```java
  public char findTheDifference(String s, String t) {
        int as = 0, at = 0;
        for (int i = 0; i < s.length(); ++i) {
            as += s.charAt(i);
        }
        for (int i = 0; i < t.length(); ++i) {
            at += t.charAt(i);
        }
        return (char) (at - as);
    }

```



**还有方法三位运算**



#### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

难度简单1761收藏分享切换为英文接收动态反馈

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

图示两个链表在节点 `c1` 开始相交**：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

**自定义评测：**

**评测系统** 的输入如下（你设计的程序 **不适用** 此输入）：

- `intersectVal` - 相交的起始节点的值。如果不存在相交节点，这一值为 `0`
- `listA` - 第一个链表
- `listB` - 第二个链表
- `skipA` - 在 `listA` 中（从头节点开始）跳到交叉节点的节点数
- `skipB` - 在 `listB` 中（从头节点开始）跳到交叉节点的节点数

评测系统将根据这些输入创建链式数据结构，并将两个头节点 `headA` 和 `headB` 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 **视作正确答案** 。

 

**示例 1：**

[![img](https://assets.leetcode.com/uploads/2021/03/05/160_example_1_1.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
```

**示例 2：**

[![img](https://assets.leetcode.com/uploads/2021/03/05/160_example_2.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)

```
输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
```

**示例 3：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)

```
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。
```

 

**提示：**

- `listA` 中节点数目为 `m`
- `listB` 中节点数目为 `n`
- `1 <= m, n <= 3 * 104`
- `1 <= Node.val <= 105`
- `0 <= skipA <= m`
- `0 <= skipB <= n`
- 如果 `listA` 和 `listB` 没有交点，`intersectVal` 为 `0`
- 如果 `listA` 和 `listB` 有交点，`intersectVal == listA[skipA] == listB[skipB]`

 

**进阶：**你能否设计一个时间复杂度 `O(m + n)` 、仅用 `O(1)` 内存的解决方案？

通过次数508,873

提交次数807,723

##### 解法一：栈

容易想到的第一个解法是通过set方法，还有一种比较容易想到的方法是通过栈的方式，按照题目描述，如果两个链表相交，那么从相交的那个节点起，这两个链表后续的节点都是相同的。自然的解法就是将两个链表节点遍历，存放到栈里，然后取出做比对。

~~~java
 public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Stack<ListNode> s1=new Stack();
        Stack<ListNode> s2=new Stack();
        while(headA!=null||headB!=null){
            if(headA!=null){
                s1.push(headA);
                headA=headA.next;
            }
            if(headB!=null){
                s2.push(headB);
                headB=headB.next;
            }
        }
        ListNode ans=null;
        while(!s1.isEmpty()&&!s2.isEmpty()){
            if(s1.peek()==s2.peek()){
                ans=s1.peek();
            }
            s1.pop();
            s2.pop();
        }
        return ans;
    }
~~~



##### 解法二：截短比较

解法一还是用到了额外的空间，考虑将两个节点的长度拉倒相同。

~~~java
int getListLength(ListNode node){
        if(node==null){
            return 0;
        }
        return 1+getListLength(node.next);
    }
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            int al=getListLength(headA);
            int bl=getListLength(headB);
            if(al<bl){
                ListNode tmp=headA;
                headA=headB;
                headB=tmp;
                int tmpLength=al;
                al=bl;
                bl=tmpLength;
            }
            //cut to same length
            ListNode pa=headA,pb=headB;
            int diffLength=al-bl;
            while(diffLength-->0){
                pa=pa.next;
            }
            while(pa!=null){
                if(pa==pb){
                    return pa;
                }
                pa=pa.next;
                pb=pb.next;
            }
            return null;
    }
~~~



#### [415. 字符串相加](https://leetcode.cn/problems/add-strings/)

难度简单586收藏分享切换为英文接收动态反馈

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 `BigInteger`）， 也不能直接将输入的字符串转换为整数形式。

 

**示例 1：**

```
输入：num1 = "11", num2 = "123"
输出："134"
```

**示例 2：**

```
输入：num1 = "456", num2 = "77"
输出："533"
```

**示例 3：**

```
输入：num1 = "0", num2 = "0"
输出："0"
```

 

 

**提示：**

- `1 <= num1.length, num2.length <= 104`
- `num1` 和`num2` 都只包含数字 `0-9`
- `num1` 和`num2` 都不包含任何前导零

通过次数216,336

提交次数393,828

##### 我的解法

直接做法，字符串相加，由于是从高位到低位，基本做法就是从低位做拼接，为了提高效率，使用字符数组。

~~~java
 void addChar(char[] num1,char[] num2,int pos,int carry,char[] ans){
        int sum=carry;
        if(pos>=ans.length){
            return;
        }
        int n1l=num1.length,n2l=num2.length;
        int ansL=ans.length;
        if(pos<num1.length){
            sum+=num1[n1l-1-pos]-'0';
        }
        if(pos<num2.length){
            sum+=num2[n2l-1-pos]-'0';
        }
        ans[ansL-1-pos]=(char)(sum%10+'0');
        addChar(num1,num2,pos+1,sum>9?1:0,ans);
    }
    public String addStrings(String num1, String num2) {
        int maxLength=Math.max(num1.length(),num2.length());
        char[] ans=new char[maxLength+1];
        Arrays.fill(ans,'0');
        addChar(num1.toCharArray(),num2.toCharArray(),0,0,ans);
        int start=0;
        if(ans[start]=='0'){
            start++;
        }else{
            maxLength++;
        }
        String result=new String(ans,start,maxLength);
        return result;
    }
~~~



##### 官方精简写法

~~~java
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
        // 计算完以后的答案需要翻转过来
        ans.reverse();
        return ans.toString();
    }
~~~



#### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

难度中等1761收藏分享切换为英文接收动态反馈

给你一个整数数组 `prices` ，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回 *你能获得的 **最大** 利润* 。

 

**示例 1：**

```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。
```

**示例 2：**

```
输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     总利润为 4 。
```

**示例 3：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 交易无法获得正利润，所以不参与交易可以获得最大利润，最大利润为 0 。
```

 

**提示：**

- `1 <= prices.length <= 3 * 104`
- `0 <= prices[i] <= 104`

通过次数682,426

提交次数964,189

##### 直接计算

允许多次买入卖出，从曲线图来看，应该在波谷时买入，紧接着在下一个下降点之前卖出，注意最后一个数值。

~~~java
  public int maxProfit(int[] prices) {
        int profit=0;
        int valley=prices[0];
        for(int i=0;i<prices.length;i++){
            valley=Math.min(valley,prices[i]);
            if(i==prices.length-1){
                profit+=prices[i]-valley;
            }
            //下降点卖出
            if(i+1<prices.length&&prices[i+1]<prices[i]){
                //sell
                profit+=prices[i]-valley;
                valley=prices[i+1];
                i++;
            }
        }
        return profit;
    }
~~~



##### 方法二：动态规划

定义$dp[i][0]$表示第$i$天交易完成后没有股票的收益，$dp[i][1]$表示第$i$天交易完成后还持有一张股票的最大收益。那么对应的动态转移方程如下：

交易完成后没有股票的最大收益
$$
dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i])
$$


交易完成后持有一张股票的最大收益
$$
dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i])
$$

~~~java
public int maxProfit(int[] prices) {
        int n = prices.length;
        int dp0 = 0, dp1 = -prices[0];
        for (int i = 1; i < n; ++i) {
            int newDp0 = Math.max(dp0, dp1 + prices[i]);
            int newDp1 = Math.max(dp1, dp0 - prices[i]);
            dp0 = newDp0;
            dp1 = newDp1;
        }
        return dp0;
    }
~~~



##### 官方贪心算法

其实跟方法一相似

~~~java
  public int maxProfit(int[] prices) {
        int ans = 0;
        int n = prices.length;
        for (int i = 1; i < n; ++i) {
            ans += Math.max(0, prices[i] - prices[i - 1]);
        }
        return ans;
    }
~~~



#### [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)

难度简单963收藏分享切换为英文接收动态反馈

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

```
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
```

**示例 2：**

```
输入：head = [], val = 1
输出：[]
```

**示例 3：**

```
输入：head = [7,7,7,7], val = 7
输出：[]
```

 

**提示：**

- 列表中的节点数目在范围 `[0, 104]` 内
- `1 <= Node.val <= 50`
- `0 <= val <= 50`

通过次数389,845

提交次数723,145

~~~java
 public ListNode removeElements(ListNode head, int val) {
        ListNode dummyHead=new ListNode();
        ListNode p=dummyHead,cur=head;
        ListNode next=null;
        while(cur!=null){
            next=cur.next;
            //set cur next null
            cur.next=null;
            if(cur.val!=val){
                p.next=cur;
                p=cur;
            }
            cur=next;
        }
        return dummyHead.next;
    }
~~~



#### [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

难度简单896收藏分享切换为英文接收动态反馈

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**示例 1：**

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

 

**限制：**

```
2 <= n <= 100000
```

通过次数598,495

提交次数881,196

##### 桶排序算法

~~~java
  public int findRepeatNumber(int[] nums) {
        int len=nums.length;
        boolean[] repeat=new boolean[len];
        Arrays.fill(repeat,false);
        for(int num:nums){
            if(repeat[num]){
                return num;
            }
            repeat[num]=true;
        }
        return -1;
    }
~~~



#### [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

难度中等1071收藏分享切换为英文接收动态反馈

编写一个高效的算法来搜索 `*m* x *n*` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg)

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid.jpg)

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= n, m <= 300`
- `-109 <= matrix[i][j] <= 109`
- 每行的所有元素从左到右升序排列
- 每列的所有元素从上到下升序排列
- `-109 <= target <= 109`

通过次数292,774

提交次数566,616

##### 横向二分查找

只是利用行的有序性来进行二分查找

~~~java
   int binarySearch(int[] nums,int low,int high,int target){
        while(low<=high){
            int mid=(low+high)/2;
            if(target==nums[mid]){
                return mid;
            }else if(target>nums[mid]){
                low=mid+1;
            }else{
                high=mid-1;
            }
        }
        return -1;
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int m=matrix.length,n=matrix[0].length;
        for(int i=0;i<m;i++){
            int index=binarySearch(matrix[i],0,n-1,target);
            if(index!=-1){
                return true;
            }
        }
        return false;
    }
}
~~~



##### 官方：Z字形查找

构建一个查找矩阵，查找矩阵的两个边界分别是左下角和右上角，查找过程的坐标为$(x,y)$(初始时刻为$(0,n-1)$)，对应的左下角坐标是

$(m-1,0)$，比较当前值$target$和$(x,y)$

1. $target$大，由于横向是升序排列的，那么$x$行的所有值都比$target$小，所以下移一行
2. $target$小，那么这一列$y$的值都比$target$大，所以要左移一列

~~~java
 public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
        }
        return false;
    }
~~~



#### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

难度简单790收藏分享切换为英文接收动态反馈

给定一个非负整数 *`numRows`，*生成「杨辉三角」的前 *`numRows`* 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

![img](https://pic.leetcode-cn.com/1626927345-DZmfxB-PascalTriangleAnimated2.gif)

 

**示例 1:**

```
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

**示例 2:**

```
输入: numRows = 1
输出: [[1]]
```

 

**提示:**

- `1 <= numRows <= 30`

通过次数322,905

提交次数429,873

##### 直接求解

~~~java
 public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans=new ArrayList();
        for(int i=0;i<numRows;i++){
            int[] row=new int[i+1];
            int len=row.length;
            for(int j=0;j<=(len-1)/2;j++){
                if(j==0){
                    row[j]=1;
                    row[len-1-j]=1;
                    continue;
                }
                //next fill
                row[j]=ans.get(i-1).get(j-1)+ans.get(i-1).get(j);
                row[len-1-j]=row[j];
            }
            ans.add(Arrays.stream(row).boxed().toList());
        }
        return ans;
    }
~~~



#### [221. 最大正方形](https://leetcode.cn/problems/maximal-square/)

难度中等1205收藏分享切换为英文接收动态反馈

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/26/max1grid.jpg)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/11/26/max2grid.jpg)

```
输入：matrix = [["0","1"],["1","0"]]
输出：1
```

**示例 3：**

```
输入：matrix = [["0"]]
输出：0
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 300`
- `matrix[i][j]` 为 `'0'` 或 `'1'`

通过次数211,813

提交次数431,809



##### 对角线解法

由于求解的是最大正方形，根据正方形的定义，求解矩阵中每一个以为当前坐标作为正方形左上位置时候的右方和下方最大连续1的个数，然后沿着这个坐标$(x,y)$作为起点，以对角线路径$(x+1,y+1)$来进行遍历，不断收缩正方形边大小，注意终止条件。

~~~java
 public int maximalSquare(char[][] matrix) {
        //trace rectangle left top position to right and bottom max distance
        int m=matrix.length,n=matrix[0].length;
        int[][] rightDistMatrix=new int[m][n];
        int[][] bottomDistMatrix=new int[m][n];
        int i,j,k;
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                //max right distance
                int rightDistance=0;
                k=j;
                while(k<n&&matrix[i][k++]=='1'){
                    rightDistance++;
                }
                rightDistMatrix[i][j]=rightDistance;
                //max bottom distance
                int bottomDistance=0;
                k=i;
                while(k<m&&matrix[k++][j]=='1'){
                    bottomDistance++;
                }
                bottomDistMatrix[i][j]=bottomDistance;
            }
        }
        //scan max matrix along triangle anxis
        int maxMatrix=0;
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                int ans=Math.min(rightDistMatrix[i][j],bottomDistMatrix[i][j]);
                k=1;
                while(i+k<m&&j+k<n&&ans>k){
                    int currentMaxDist=Math.min(rightDistMatrix[i+k][j+k],
                            bottomDistMatrix[i+k][j+k]);
                    ans=Math.min(ans,currentMaxDist+k);
                    if(currentMaxDist==0){
                        break;
                    }
                    k++;
                }
                maxMatrix=Math.max(maxMatrix,ans);
            }
        }
        return maxMatrix*maxMatrix;
    }
~~~



##### 官方：动态规划

使用$dp[i][j]$表示以$(i,j)$作为正方形右下角的最大正方形面积，那么得到对应的动态规划转移方程，证明相见

[统计全为 1 的正方形子矩阵]: https://leetcode.cn/problems/count-square-submatrices-with-all-ones/solution/tong-ji-quan-wei-1-de-zheng-fang-xing-zi-ju-zhen-2/


$$
dp[i][j]=Math.min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])+1
前提是matrix[i][j]=0
$$
代码如下：

~~~java
 public int maximalSquare(char[][] matrix) {
        int maxSide = 0;
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return maxSide;
        }
        int rows = matrix.length, columns = matrix[0].length;
        int[][] dp = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxSide = Math.max(maxSide, dp[i][j]);
                }
            }
        }
        int maxSquare = maxSide * maxSide;
        return maxSquare;
    }
~~~



#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

难度中等2064收藏分享切换为英文接收动态反馈

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 *所有* **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

 

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

**示例 2：**

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```
输入: candidates = [2], target = 1
输出: []
```

 

**提示：**

- `1 <= candidates.length <= 30`
- `1 <= candidates[i] <= 200`
- `candidate` 中的每个元素都 **互不相同**
- `1 <= target <= 500`

通过次数551,008

提交次数758,045



##### 方法一：递归求解

dfs和回溯过程，递归里面的流程是当前节点是否要添加进去

~~~java
List<List<Integer>> ans=new ArrayList();
    void dfs(int[] candidates,int target,int index,ArrayList<Integer> result){
        if(target==0){
            ans.add(new ArrayList(result));
            return;
        }
        if(index==candidates.length||target<candidates[index]){
            return;
        }
        //minus
        result.add(candidates[index]);
        dfs(candidates,target-candidates[index],index,result);
        result.remove(result.size()-1);
         //skip current
        dfs(candidates,target,index+1,result);

    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        dfs(candidates,target,0,new ArrayList());
        return ans;
    }
~~~



##### 方法二：官方解答

常规的trace过程

~~~java
 List<List<Integer>> pathList=new ArrayList();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        backtrace(candidates,target,0,0,new ArrayList());
        return pathList;
    }
    void backtrace(int[] candidates,int target,int sum,int startIndex,List<Integer> path) {
        if(sum==target){
            pathList.add(new ArrayList(path));
            return;
        }
        for(int i=startIndex;i<candidates.length;i++){
            if(sum+candidates[i]<=target){
                path.add(candidates[i]);
                backtrace(candidates,target,sum+candidates[i],i,path);
                path.remove(path.size()-1);
            }            
        }
    }
~~~



#### [191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/)

难度简单489收藏分享切换为英文接收动态反馈

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为[汉明重量](https://baike.baidu.com/item/汉明重量)）。

 

**提示：**

- 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
- 在 Java 中，编译器使用[二进制补码](https://baike.baidu.com/item/二进制补码/5295284)记法来表示有符号整数。因此，在上面的 **示例 3** 中，输入表示有符号整数 `-3`。

 

**示例 1：**

```
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

**示例 2：**

```
输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```

**示例 3：**

```
输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
```

 

**提示：**

- 输入必须是长度为 `32` 的 **二进制串** 。



 

**进阶**：

- 如果多次调用这个函数，你将如何优化你的算法？

通过次数267,905

提交次数352,677

##### 方法一：转换取反

我的解法，输入数值如果是非负数直接按照位移计算累加即可，如果是负数，实际上计算的是这个负数对应的补码对应的绝对值的1的个数。将其转换得到该补码对应的绝对值，为了避免整数位溢出，转换为$long$类型，转换公式**(还要添加符号位的一个1)**
$$
pos=Integer.MAX\_VALUE+1L+n
$$


~~~java
// you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        long pos=n;
        boolean isNeg=false;
        //if negative, get corresponding positive
        if(n<0){
            isNeg=true;
            pos=Integer.MAX_VALUE+1L+n;
        }
        int oneCount=isNeg?1:0;
        while(pos>0){
            if((pos&1)==1){
                oneCount++;
            }
            pos>>=1;
        }
        return oneCount;
    }
~~~



##### 官方解法二：算数右移

直接利用有符号算数右移即可

~~~java
 public int hammingWeight(int n) {
        int ret = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & (1 << i)) != 0) {
                ret++;
            }
        }
        return ret;
    }
~~~



##### 官方解法二：位运算优化

利用性质$n\&(n-1)$的值把 $n$ 的二进制位中的最低位的 $1 $变为 $0$ 之后的结果性质，不断的转换$n$，直到当前为$0$，循环终止。

~~~java
public int hammingWeight(int n) {
        int ret = 0;
        while (n != 0) {
            n &= n - 1;
            ret++;
        }
        return ret;
    }
~~~



#### [179. 最大数](https://leetcode.cn/problems/largest-number/)

难度中等970收藏分享切换为英文接收动态反馈

给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

**注意：**输出结果可能非常大，所以你需要返回一个字符串而不是整数。

 

**示例 1：**

```
输入：nums = [10,2]
输出："210"
```

**示例 2：**

```
输入：nums = [3,30,34,5,9]
输出："9534330"
```

 

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 109`

通过次数164,368

提交次数399,449

##### 我的解法

直接是字符串排序，注意数值都是$0$的情景。

~~~java
 int compareLexical(Integer num1,Integer num2){
        String s1=String.valueOf(num1);
        String s2=String.valueOf(num2);
        String beforeS1=s1+s2;
        String beforeS2=s2+s1;
        return beforeS2.compareTo(beforeS1); 
    }
    public String largestNumber(int[] nums) {
        Comparator<Integer> comparator=this::compareLexical;
        List<Integer> numList=Arrays.stream(nums).boxed().collect(Collectors.toList());
        numList.sort(comparator);
        StringBuilder ans=new StringBuilder();
        boolean empty=true;
        for(Integer num:numList){
            if(num==0&&empty){
                continue;
            }
            ans.append(num);
            empty=false;
        }
        
        return empty?"0":ans.toString();
    }
~~~



##### 官方解答：数值排序

没有采用字符串拼接比较的方式，而是直接使用数值比较。其中$sy*x+y$表示$y,x$的拼接结果,$sx*y+x$表示$x,y$的拼接结果。

~~~java
 public String largestNumber(int[] nums) {
        int n = nums.length;
        // 转换成包装类型，以便传入 Comparator 对象（此处为 lambda 表达式）
        Integer[] numsArr = new Integer[n];
        for (int i = 0; i < n; i++) {
            numsArr[i] = nums[i];
        }

        Arrays.sort(numsArr, (x, y) -> {
            long sx = 10, sy = 10;
            while (sx <= x) {
                sx *= 10;
            }
            while (sy <= y) {
                sy *= 10;
            }
            return (int) (-sy * x - y + sx * y + x);
        });

        if (numsArr[0] == 0) {
            return "0";
        }
        StringBuilder ret = new StringBuilder();
        for (int num : numsArr) {
            ret.append(num);
        }
        return ret.toString();
    }
~~~



#### [1116. 打印零与奇偶数](https://leetcode.cn/problems/print-zero-even-odd/)

难度中等133收藏分享切换为英文接收动态反馈

现有函数 `printNumber` 可以用一个整数参数调用，并输出该整数到控制台。

- 例如，调用 `printNumber(7)` 将会输出 `7` 到控制台。

给你类 `ZeroEvenOdd` 的一个实例，该类中有三个函数：`zero`、`even` 和 `odd` 。`ZeroEvenOdd` 的相同实例将会传递给三个不同线程：

- **线程 A：**调用 `zero()` ，只输出 `0`
- **线程 B：**调用 `even()` ，只输出偶数
- **线程 C：**调用 `odd()` ，只输出奇数

修改给出的类，以输出序列 `"010203040506..."` ，其中序列的长度必须为 `2n` 。

实现 `ZeroEvenOdd` 类：

- `ZeroEvenOdd(int n)` 用数字 `n` 初始化对象，表示需要输出的数。
- `void zero(printNumber)` 调用 `printNumber` 以输出一个 0 。
- `void even(printNumber)` 调用`printNumber` 以输出偶数。
- `void odd(printNumber)` 调用 `printNumber` 以输出奇数。

 

**示例 1：**

```
输入：n = 2
输出："0102"
解释：三条线程异步执行，其中一个调用 zero()，另一个线程调用 even()，最后一个线程调用odd()。正确的输出为 "0102"。
```

**示例 2：**

```
输入：n = 5
输出："0102030405"
```

 

**提示：**

- `1 <= n <= 1000`

通过次数28,519

提交次数53,667

##### 我的解法

利用$ReentrantLock$和$Condition$，针对当前输入的数值来唤醒不同的线程，注意最后的终止条件以及$0,1$这种输出结果，输出的时候要求所有线程都不能再lock中。(位于$\&$运算符的优先级低于算数运算符)。

~~~java
class ZeroEvenOdd {
    private int n;
    private int totalNumber;
    private ReentrantLock lock = new ReentrantLock();
    private Condition zeroCondition = lock.newCondition();
    private Condition oddCondition = lock.newCondition();
    private Condition evenCondition = lock.newCondition();

    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
        while (totalNumber < 2 * n) {
            lock.lock();
            int next = totalNumber + 1;
            if (next % 4 == 0) {
                evenCondition.signal();
                zeroCondition.await();
            } else if (next % 2 == 2) {
                oddCondition.signal();
                zeroCondition.await();
            } else {
                totalNumber++;
                printNumber.accept(0);
                next = totalNumber + 1;
                if (next % 4 == 0) {
                    evenCondition.signal();
                } else {
                    oddCondition.signal();
                }
                if (totalNumber != 2 * n) {
                    zeroCondition.await();
                }
            }
            lock.unlock();
        }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
        while (totalNumber < 2 * n) {
            lock.lock();
            int next = totalNumber + 1;
            if (next % 4 != 0) {
                evenCondition.await();
            }
            if(totalNumber==2*n){
                break;
            }
            totalNumber++;
            printNumber.accept(totalNumber >> 1);
            zeroCondition.signal();
            if (totalNumber != 2 * n) {
                evenCondition.await();
            }else{
                oddCondition.signal();
            }
            lock.unlock();
        }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
        while (totalNumber < 2 * n) {
            lock.lock();
            int next = totalNumber + 1;
            if (next % 4 == 0 || (next & 1) == 1) {
                oddCondition.await();
            }
            if(totalNumber==2*n){
                break;
            }
            totalNumber++;
            printNumber.accept(totalNumber >> 1);
            zeroCondition.signal();
            if (totalNumber != 2 * n) {
                oddCondition.await();
            }else{
                evenCondition.signal();
            }
            lock.unlock();
        }
    }
}
~~~



##### synchronized方法

~~~java
class ZeroEvenOdd {
    private int n;
    private volatile int flag = 1;
    
    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
        for (int i=1; i<=n; i++) {
            synchronized (this) {
                while (flag % 2 == 0) wait();
                printNumber.accept(0);
                flag ++;
                notifyAll();
            }
        }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
        for (int i=2; i<=n; i+=2) {
            synchronized (this) {
                while (flag % 4 != 0) wait();
                printNumber.accept(i);
                flag ++;
                notifyAll();
            }
        }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
        for (int i=1; i<=n; i+=2) {
            synchronized(this) {
                while (flag % 4 != 2) wait();
                printNumber.accept(i);
                flag ++;
                notifyAll();
            }
        }
    }
}
~~~



#### [912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

难度中等611收藏分享切换为英文接收动态反馈

给你一个整数数组 `nums`，请你将该数组升序排列。

 



**示例 1：**

```
输入：nums = [5,2,3,1]
输出：[1,2,3,5]
```

**示例 2：**

```
输入：nums = [5,1,1,2,0,0]
输出：[0,0,1,1,2,5]
```

 

**提示：**

- `1 <= nums.length <= 5 * 104`
- `-5 * 104 <= nums[i] <= 5 * 104`

通过次数404,991

提交次数727,423

###### 我的解法：混合排序

根据当前数组的大小进行选择排序，如果数组数量小的话 ，直接使用比较排序；否则使用桶排序。

~~~java
  public int[] sortArray(int[] nums) {
        if(nums.length<=1000){
            Arrays.sort(nums);
            return nums;
        }
        int posMax=5*10000;
        int[] fill=new int[2*posMax+1];
        for(int num:nums){
            fill[num+posMax]++;
        }
        int i=0,j=0;
        for(;i<fill.length;i++){
            int count=fill[i];
            if(count==0){
                continue;
            }
            while(count-->0){
                nums[j++]=i-posMax;
            }
        }
        return nums;

    }
~~~



##### 官方：随机快排

~~~java
 public int[] sortArray(int[] nums) {
        randomizedQuicksort(nums, 0, nums.length - 1);
        return nums;
    }

    public void randomizedQuicksort(int[] nums, int l, int r) {
        if (l < r) {
            int pos = randomizedPartition(nums, l, r);
            randomizedQuicksort(nums, l, pos - 1);
            randomizedQuicksort(nums, pos + 1, r);
        }
    }

    public int randomizedPartition(int[] nums, int l, int r) {
        int i = new Random().nextInt(r - l + 1) + l; // 随机选一个作为我们的主元
        swap(nums, r, i);
        return partition(nums, l, r);
    }

    public int partition(int[] nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums, i, j);
            }
        }
        swap(nums, i + 1, r);
        return i + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
~~~



#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

难度中等1807收藏分享切换为英文接收动态反馈

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

 

**提示：**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`
- `nums` 是一个非递减数组
- `-109 <= target <= 109`

通过次数589,578

提交次数1,393,534

##### 我的解法

直接二分查找变形

~~~java
 int binarySearch(int [] nums,int target , int direction){
        int pos=-1;
        int left=0,right=nums.length-1;
        while(left<=right){
            int mid=(left+right)/2;
            if(nums[mid]>target){
                right=mid-1;
            }else if(nums[mid]<target){
                left=mid+1;
            }else{
                pos=mid;
                if(direction==0){
                    right=mid-1;
                }else{
                    left=mid+1;
                }
            }
        }
        return pos;
    }
    public int[] searchRange(int[] nums, int target) {
        int leftBound=binarySearch(nums,target,0);
        int rightBound=binarySearch(nums,target,1);
        return new int[]{leftBound,rightBound};
    }
~~~



#### [27. 移除元素](https://leetcode.cn/problems/remove-element/)

难度简单1390收藏分享切换为英文接收动态反馈

给你一个数组 `nums` 和一个值 `val`，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 `val` 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 `O(1)` 额外空间并 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组**。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

 

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
int len = removeElement(nums, val);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

 

**示例 1：**

```
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
```

**示例 2：**

```
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。
```

 

**提示：**

- `0 <= nums.length <= 100`
- `0 <= nums[i] <= 50`
- `0 <= val <= 100`

通过次数759,428

提交次数1,276,758

##### 快排数值替换

类似于快排中的数值替换，使用双指针，一个表示当前遍历的索引，一个表示不是目标值的索引。

~~~java
 void swapInt(int[] nums,int i,int j){
        int tmp=nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
    }
    public int removeElement(int[] nums, int val) {
        int i=0,j=-1;
        int len=nums.length;
        for(;i<len;i++){
            if(nums[i]!=val){
                j++;
                swapInt(nums,i,j);
            }
        }
        return j+1;
    }
~~~



##### 官方解法

注意$right-1$才是空出来的位置

~~~java
 public int removeElement(int[] nums, int val) {
        int left = 0;
        int right = nums.length;
        while (left < right) {
            if (nums[left] == val) {
                nums[left] = nums[right - 1];
                right--;
            } else {
                left++;
            }
        }
        return left;
    }
~~~



#### [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

难度中等1820收藏分享切换为英文接收动态反馈

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须**[ 原地 ](https://baike.baidu.com/item/原地算法)**修改，只允许使用额外常数空间。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

**示例 3：**

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

 

**提示：**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 100`

通过次数334,662

提交次数885,068

##### 我的解法

题目开始时候没有思路，看解析才有想法。从数值的角度出来，要想找到下一个序列，就是找一个数值：

- 数值位增加的位尽量在右边，这样，能够最小程度的增大
- 这个增加的数值位的值要尽可能比原先大一点

比如对于数值$436521$来说，从右向左出发找到一个下降的序列对$36$，下一次要增加的序列为就是$3$，用什么值来替换它，用比它大并且尽可能小的数，由于$3$后面是一个递减（包含相等）$6521$序列，所以查找到该路径下比它大的第一个值$5$，然后替换，替换后位置$3$之后的序列是$6321$，还要进行反转。

~~~java
void revert(int[] nums,int left,int right){
        int limit=(right-left+1)/2+left;
        for(int i=left;i<limit;i++){
            int tmp=nums[i];
            nums[i]=nums[right+left-i];
            nums[right+left-i]=tmp;
        }
    }
    public void nextPermutation(int[] nums) {
        //从右向左找到相邻的降序的序列组
        int i,j;
        boolean found=false;
        for(i=nums.length-1;i>0;i--){
            if(nums[i-1]<nums[i]){
                found=true;
                break;
            }
        }
        //当前已经是最大序列，直接反转数组
        if(!found){
            revert(nums,0,nums.length-1);
            return;
        }
        //从右向左找到比nums[i-1]第一个大的数，然后交换
        j=nums.length-1;
        while(nums[j]<=nums[i-1]){
            j--;
        }
        //交换
        int tmp=nums[j];
        nums[j]=nums[i-1];
        nums[i-1]=tmp;
        //排序i到length-1,此时顺序是递增的，只需反转即可
        revert(nums,i,nums.length-1);
    }
~~~



##### 官方答案

~~~java
 public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }
~~~



#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

难度困难1753收藏分享切换为英文接收动态反馈

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

 

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```
输入：nums = [1], k = 1
输出：[1]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`
- `1 <= k <= nums.length`

通过次数325,863

提交次数652,002

##### 方法一：堆排序

堆中维护的是一个滑动窗口大小左右的数值以及对应的位置，窗口移动的时候，每次获取当前堆中的最大值，比较当前值是否还在滑动窗口内，如果不在那么移除。

~~~java
/**
     * 基本思路：要获取滑动窗口的最大值，自然想到堆求解，考虑到随着滑动窗口的更新，如果
     * 堆中的数据不在当前窗口内，那么应该删除。堆内容存储的是当前数值和其位置，每次获取
     * 堆顶部数值，如果不在当前窗口内，删除即可
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        //两个数值数组，位置0表示数值，位置1表示索引值
        Comparator<int[]> bigComparator=(n1,n2)->n2[0]-n1[0];
        int[] ans=new int[nums.length-k+1];
        int j=0;
        PriorityQueue<int[]> queue=new PriorityQueue(bigComparator);
        for(int i=0;i<nums.length;i++){
            int[] wrapNum=new int[]{nums[i],i};
            queue.offer(wrapNum);
            //删除过期的数值
            if(queue.size()>k){
                int leftBound=i-k+1;
                while(!queue.isEmpty()){
                    int[] top=queue.peek();
                    if(top[1]<leftBound){
                        queue.poll();
                    }else{
                        break;
                    }
                }
            }
            if(i>=k-1){
                ans[j++]=queue.peek()[0];
            }
        }
        return ans;
    }
~~~



##### 官方题解：优先队列

~~~java
 public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] pair1, int[] pair2) {
                return pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair2[1] - pair1[1];
            }
        });
        for (int i = 0; i < k; ++i) {
            pq.offer(new int[]{nums[i], i});
        }
        int[] ans = new int[n - k + 1];
        ans[0] = pq.peek()[0];
        for (int i = k; i < n; ++i) {
            pq.offer(new int[]{nums[i], i});
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
        }
        return ans;
    }

~~~



##### 官方题解2：单调队列

对于滑动窗口中的任意两个下标$i$和$j$($i<j$)，只要满足$nums[i]\leq nums[j]$,那么滑动中的最大值就不会是$nums[i]$。根据此思路，可以维护一个严格递减的单调栈（使用队列实现，因为需要获取当前栈中的最大值，即第一个数值），顺序比较当前值和栈顶值，如果当前值大于或者等于栈顶值，那么出栈，当前值入栈。

~~~java
 public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        Deque<Integer> deque = new LinkedList<Integer>();
        for (int i = 0; i < k; ++i) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
        }

        int[] ans = new int[n - k + 1];
        ans[0] = nums[deque.peekFirst()];
        for (int i = k; i < n; ++i) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            while (deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }
            ans[i - k + 1] = nums[deque.peekFirst()];
        }
        return ans;
    }
~~~



#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

难度困难3634收藏分享切换为英文接收动态反馈

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

**示例 2：**

```
输入：height = [4,2,0,3,2,5]
输出：9
```

 

**提示：**

- `n == height.length`
- `1 <= n <= 2 * 104`
- `0 <= height[i] <= 105`

通过次数531,237

提交次数862,692

##### 我的解法：更新边界

计算以当前柱子高度能够存放的雨水，就是从当前柱子下标开始向左和向右移动找到最大的值。

~~~java
/**
     * 接多少雨水的条件是：以当前柱子i为准，向左、右边界不断扩展，找到左右边界的最大值，积累的雨水数量就是
     * 差值
     */
    public int trap(int[] height) {
        int[] leftMax=new int[height.length];
        int[] rightMax=new int[height.length];
        //计算左边界
        int i,max=height[0];
        for(i=0;i<height.length;i++){
            max=Math.max(height[i],max);
            leftMax[i]=max;
        }
        //计算右边界
        i=height.length-1;
        max=height[i];
        while(i>=0){
            max=Math.max(max,height[i]);
            rightMax[i]=max;
            i--;
        }
        //计算雨水
        int sum=0;
        for(i=0;i<height.length;i++){
            sum+=Math.min(leftMax[i],rightMax[i])-height[i];
        }
        return sum;
    }
~~~



##### 官方题解：单调栈

除了计算并存储每个位置两边的最大高度以外，也可以用单调栈计算能接的雨水总量。

维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 $height$ 中的元素递减。

从左到右遍历数组，遍历到下标 ii 时，如果栈内至少有两个元素，记栈顶元素为 $top$ 的下面一个元素是 $left$，则一定有 

$height[top] \leq height[left] $。如果 $height[i] \geq heigth[top]$，则得到一个可以接雨水的区域，该区域的宽度是 $i-left-1$，高度是 $min(height[left],height[i])-height[top]$，根据宽度和高度即可计算得到该区域能接的雨水量，**注意这个积水区域值得是高度值。**

~~~java
public int trap(int[] height) {
        int ans = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        int n = height.length;
        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int left = stack.peek();
                int currWidth = i - left - 1;
                int currHeight = Math.min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stack.push(i);
        }
        return ans;
    }

~~~



##### 官方解答：双指针

~~~java
 public int trap(int[] height) {
        int ans = 0;
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (height[left] < height[right]) {
                ans += leftMax - height[left];
                ++left;
            } else {
                ans += rightMax - height[right];
                --right;
            }
        }
        return ans;
    }

~~~



#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

难度中等2075收藏分享切换为英文接收动态反馈

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 *所有* **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

 

**示例 1：**

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

**示例 2：**

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

**示例 3：**

```
输入: candidates = [2], target = 1
输出: []
```

 

**提示：**

- `1 <= candidates.length <= 30`
- `1 <= candidates[i] <= 200`
- `candidate` 中的每个元素都 **互不相同**
- `1 <= target <= 500`

通过次数557,420

提交次数766,786



##### 回溯和基本剪枝

~~~java
List<List<Integer>> ans=new ArrayList();
    void dfs(int[] candidates,int target,int index,ArrayList<Integer> result){
        if(target==0){
            ans.add(new ArrayList(result));
            return;
        }
        if(index==candidates.length||target<candidates[index]){
            return;
        }
        //minus
        result.add(candidates[index]);
        dfs(candidates,target-candidates[index],index,result);
        result.remove(result.size()-1);
         //skip current
        dfs(candidates,target,index+1,result);

    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        dfs(candidates,target,0,new ArrayList());
        return ans;
    }
}
~~~



##### 官方解答

~~~java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> combine = new ArrayList<Integer>();
        dfs(candidates, target, ans, combine, 0);
        return ans;
    }

    public void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            ans.add(new ArrayList<Integer>(combine));
            return;
        }
        // 直接跳过
        dfs(candidates, target, ans, combine, idx + 1);
        // 选择当前数
        if (target - candidates[idx] >= 0) {
            combine.add(candidates[idx]);
            dfs(candidates, target - candidates[idx], ans, combine, idx);
            combine.remove(combine.size() - 1);
        }
    }

~~~



#### [350. 两个数组的交集 II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

难度简单800收藏分享切换为英文接收动态反馈

给你两个整数数组 `nums1` 和 `nums2` ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。

 

**示例 1：**

```
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
```

**示例 2:**

```
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```

 

**提示：**

- `1 <= nums1.length, nums2.length <= 1000`
- `0 <= nums1[i], nums2[i] <= 1000`

 

***\*进阶\**：**

- 如果给定的数组已经排好序呢？你将如何优化你的算法？
- 如果 `nums1` 的大小比 `nums2` 小，哪种方法更优？
- 如果 `nums2` 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

通过次数381,480

提交次数676,387



##### 排序比较

~~~java
 public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i=0,j=0;
        List<Integer> ans=new ArrayList();
        while(i<nums1.length&&j<nums2.length){
            if(nums1[i]<nums2[j]){
                i++;
                continue;
            }else if(nums1[i]>nums2[j]){
                j++;
                continue;
            }
            ans.add(nums1[i]);
            i++;
            j++;
        }
        return ans.stream().mapToInt(Integer::intValue).toArray();
    }
~~~



##### 官方解答: map映射

**直接借鉴一点是数组交换**

~~~
 public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return intersect(nums2, nums1);
        }
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int num : nums1) {
            int count = map.getOrDefault(num, 0) + 1;
            map.put(num, count);
        }
        int[] intersection = new int[nums1.length];
        int index = 0;
        for (int num : nums2) {
            int count = map.getOrDefault(num, 0);
            if (count > 0) {
                intersection[index++] = num;
                count--;
                if (count > 0) {
                    map.put(num, count);
                } else {
                    map.remove(num);
                }
            }
        }
        return Arrays.copyOfRange(intersection, 0, index);
    }

~~~



#### [81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)

难度中等624收藏分享切换为英文接收动态反馈

已知存在一个按非降序排列的整数数组 `nums` ，数组中的值不必互不相同。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转** ，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,4,4,5,6,6,7]` 在下标 `5` 处经旋转后可能变为 `[4,5,6,6,7,0,1,2,4,4]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 `nums` 中存在这个目标值 `target` ，则返回 `true` ，否则返回 `false` 。

你必须尽可能减少整个操作步骤。

 

**示例 1：**

```
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true
```

**示例 2：**

```
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
```

 

**提示：**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- 题目数据保证 `nums` 在预先未知的某个下标上进行了旋转
- `-104 <= target <= 104`

 

**进阶：**

- 这是 [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/) 的延伸题目，本题中的 `nums` 可能包含重复元素。
- 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

 

通过次数167,150

提交次数405,134

##### 二分搜索

~~~java
 int binarySearchIndex(int[] nums,int low,int high,int target){
        while(low<=high){
            int mid=(low+high)/2;
            if(nums[mid]==target){
                return mid;
            }else if(nums[mid]>target){
                high=mid-1;
            }else{
                low=mid+1;
            }
        }
        return -1;
    }
    boolean binarySearch(int[] nums,int target,int left,int right){
        if(left==right){
            return nums[left]==target;
        }
        int mid=(left+right)/2;
        if(nums[left]<nums[mid]){
            int index=binarySearchIndex(nums,left,mid,target);
            if(index!=-1){
                return true;
            }
        }
        if(nums[mid]<nums[right]){
            int index=binarySearchIndex(nums,mid,right,target);
            if(index!=-1){
                return true;
            }
        }
        if(binarySearch(nums,target,left,mid)){
            return true;
        }
        return binarySearch(nums,target,mid+1,right);
    }
    public boolean search(int[] nums, int target) {
        return binarySearch(nums,target,0,nums.length-1);
    }
~~~





#### [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

难度中等1332收藏分享切换为英文接收动态反馈

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

 

**示例 1：**

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

**示例 2：**

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

 

**提示：**

- `0 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`

通过次数293,876

提交次数533,693



##### 上下遍历

~~~java
 public int longestConsecutive(int[] nums) {
       //通过map来实现，指定一个位置向两边扩展
       int ans=0;
       int tmpAns=0;
       Map<Integer,Boolean> visitMap=Arrays.stream(nums).boxed().collect(Collectors.toMap(Function.identity(),k->Boolean.FALSE,(v1,v2)->v1));
       for(int num:nums){
           Boolean visited=visitMap.get(num);
           if(visited){
               continue;
           }
           tmpAns=1;
           //想下扩展
           int bench=num;
           while(visitMap.containsKey(--bench)){
               tmpAns++;
               visitMap.put(bench,Boolean.TRUE);
           }
           //想上扩展
           bench=num;
           while(visitMap.containsKey(++bench)){
               tmpAns++;
               visitMap.put(bench,Boolean.TRUE);
           }
           ans=Math.max(ans,tmpAns);
       }
       return ans;
    }
~~~



##### 优化方法

~~~java
public static int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        if (nums.length == 1) {
            return 1;
        }

        Set<Integer> map = new HashSet<>(nums.length);
        for (int x : nums) {
            map.add(x);
        }

        int maxLength = 0;
        while (!map.isEmpty()) {
            // 取map的第一个值
            int x = map.iterator().next();
            map.remove(x);

            int length = 1;

            int tmp = x;
            while (map.remove(--tmp)) {
                length += 1;
            }

            tmp = x;
            while (map.remove(++tmp)) {
                length += 1;
            }

            if (length > maxLength) {
                maxLength = length;
            }
        }

        return maxLength;
    }
~~~



#### [36. 有效的数独](https://leetcode.cn/problems/valid-sudoku/)

难度中等923收藏分享切换为英文接收动态反馈

请你判断一个 `9 x 9` 的数独是否有效。只需要 **根据以下规则** ，验证已经填入的数字是否有效即可。

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）

 

**注意：**

- 一个有效的数独（部分已被填充）不一定是可解的。
- 只需要根据以上规则，验证已经填入的数字是否有效即可。
- 空白格用 `'.'` 表示。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/04/12/250px-sudoku-by-l2g-20050714svg.png)

```
输入：board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：true
```

**示例 2：**

```
输入：board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：false
解释：除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。 但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
```

 

**提示：**

- `board.length == 9`
- `board[i].length == 9`
- `board[i][j]` 是一位数字（`1-9`）或者 `'.'`

通过次数302,399

提交次数476,759



##### 常规遍历

注意处理块之间的遍历方法

~~~java
public boolean isValidSudoku(char[][] board) {
        //校验行
        int m=9,n=9;
        int i,j;
        for(i=0;i<m;i++){
            int[] flag=new int[n];
            for(j=0;j<n;j++){
                char ch=board[i][j];
                if(ch=='.'){
                    continue;
                }
                flag[ch-'1']+=1;
                if(flag[ch-'1']>1){
                    return false;
                }
            }
        }
        //检验列
        for(j=0;j<n;j++){
            int[] flag=new int[m];
            for(i=0;i<m;i++){
                char ch=board[i][j];
                if(ch=='.'){
                    continue;
                }
                flag[ch-'1']+=1;
                if(flag[ch-'1']>1){
                    return false;
                }
            }
        }
        //对角
        i=0;
        j=0;
        while(i<m&&j<n){
            int x=i,y=j;
            int[] flag=new int[m];
            for(;x<i+3;x++){
                for(y=j;y<j+3;y++){
                    char ch=board[x][y];
                    if(ch=='.'){
                        continue;
                    }
                    flag[ch-'1']+=1;
                    if(flag[ch-'1']>1){
                        return false;
                    }
                }
            }
            if(y<n-1){
                j+=3;
            }else{
                i+=3;
                j=0;
            }
        }
        return true;
    }
~~~





#### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

难度中等1369收藏分享切换为英文接收动态反馈

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**[ 原地](https://baike.baidu.com/item/原地算法)** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)

```
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

 

**提示：**

- `n == matrix.length == matrix[i].length`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

 

通过次数352,927

提交次数475,004

##### 使用额外数组

对于数组元素$matrix[i][j]$旋转90度之后对应的位置是$matrix[n-1-i][j]$

~~~java
 public void rotate(int[][] matrix) {
        int m=matrix.length,n=matrix[0].length;
        int[][] help=new int[m][n];
        int i,j;
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                help[j][n-1-i]=matrix[i][j];
            }
        }
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                matrix[i][j]=help[i][j];
            }
        }
    }
~~~



##### 官方解答：水平反转+主对角线反转

~~~java
 public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = temp;
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

~~~



#### [263. 丑数](https://leetcode.cn/problems/ugly-number/)

难度简单334收藏分享切换为英文接收动态反馈

**丑数** 就是只包含质因数 `2`、`3` 和 `5` 的正整数。

给你一个整数 `n` ，请你判断 `n` 是否为 **丑数** 。如果是，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：n = 6
输出：true
解释：6 = 2 × 3
```

**示例 2：**

```
输入：n = 1
输出：true
解释：1 没有质因数，因此它的全部质因数是 {2, 3, 5} 的空集。习惯上将其视作第一个丑数。
```

**示例 3：**

```
输入：n = 14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数 7 。
```

 

**提示：**

- `-231 <= n <= 231 - 1`

通过次数131,017

提交次数256,269

##### 直接解答

~~~java
 public boolean isUgly(int n) {
        if(n<=0){
            return false;
        }
        while(n!=1){
            if(n%2==0){
                n/=2;
            }else if(n%3==0){
                n/=3;
            }else if(n%5==0){
                n/=5;
            }else{
                return false;
            }
        }
        return true;
    }
~~~



#### [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

难度困难2066收藏分享切换为英文接收动态反馈

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

 

**提示：**

- `1 <= heights.length <=105`
- `0 <= heights[i] <= 104`

通过次数270,627

提交次数608,669

##### 单调栈

以柱状所在高度作为矩形的高度，不断向左右两边扩展，找到左右两边首个低于当前高度的柱子，这个柱子就是边界。本质上就是

找到左右两边第一个低于当前高度的柱状，作为矩形的边界（不包含），然后求解。对于任意两个高度下标$i,j(i \lt j)$，如果$height[i] \ge height[j]$，那么对于下标$k(k \lt j)$，$k$的左边界不可能是$height[i]$，由此我们想到维护一个单调递增的栈，下面代码对于右边界做了优化处理。**注意右边界需要初始化**

~~~java
 public int largestRectangleArea(int[] heights) {
        //基于单调栈的实现思路，以当前数组元素所在高度作为矩阵的高度，然后向
        //左右不断扩展边界
        int[] left=new int[heights.length];
        int[] right=new int[heights.length];
     	//注意右边界需要初始化
        Arrays.fill(right,heights.length);
        int[] stack=new int[heights.length];
        int top=-1;
        //计算左边界、右边界
        int i;
        int maxArea=0;
        for(i=0;i<heights.length;i++){
            while(top!=-1&&heights[stack[top]]>=heights[i]){
                //right[i]不一定是i的准确右边界，但是不影响最终的计算结果，严格右边界
                //是值低于当前高度，但是对于等于条件，最后一个的计算结果就是
                right[stack[top]]=i;
                top--;
            }
            left[i]=top==-1?-1:stack[top];
            stack[++top]=i;
        }
        for(i=0;i<heights.length;i++){
            maxArea=Math.max(maxArea,heights[i]*(right[i]-left[i]-1));
        }
        return maxArea;
    }
~~~



##### 官方解答：暴力破解

~~~java
int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        int ans = 0;
        for (int mid = 0; mid < n; ++mid) {
            // 枚举高
            int height = heights[mid];
            int left = mid, right = mid;
            // 确定左右边界
            while (left - 1 >= 0 && heights[left - 1] >= height) {
                --left;
            }
            while (right + 1 < n && heights[right + 1] >= height) {
                ++right;
            }
            // 计算面积
            ans = max(ans, (right - left + 1) * height);
        }
        return ans;
    }
~~~



##### 官方解法：两次遍历单调栈 

~~~java
 public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];
        
        Deque<Integer> mono_stack = new ArrayDeque<Integer>();
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.isEmpty() ? -1 : mono_stack.peek());
            mono_stack.push(i);
        }

        mono_stack.clear();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.isEmpty() ? n : mono_stack.peek());
            mono_stack.push(i);
        }
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = Math.max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
~~~



#### [143. 重排链表](https://leetcode.cn/problems/reorder-list/)

难度中等968收藏分享切换为英文接收动态反馈

给定一个单链表 `L` 的头节点 `head` ，单链表 `L` 表示为：

```
L0 → L1 → … → Ln - 1 → Ln
```

请将其重新排列后变为：

```
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
```

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

**示例 1：**

![img](https://pic.leetcode-cn.com/1626420311-PkUiGI-image.png)

```
输入：head = [1,2,3,4]
输出：[1,4,2,3]
```

**示例 2：**

![img](https://pic.leetcode-cn.com/1626420320-YUiulT-image.png)

```
输入：head = [1,2,3,4,5]
输出：[1,5,2,4,3]
```

 

**提示：**

- 链表的长度范围为 `[1, 5 * 104]`
- `1 <= node.val <= 1000`

通过次数196,486

提交次数306,576

##### 我的解法：链表二分

~~~java
ListNode getMidNode(ListNode head){
        if(head==null||head.next==null){
            return head;
        }
        ListNode slow=head,fast=head.next;
        while(fast.next!=null&&fast.next.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        //断链
        ListNode mid;
        if(fast.next==null){
            mid=slow.next;
            slow.next=null;
        }else{
            mid=slow.next.next;
            slow.next.next=null;
        }
        //fast.next==null表明当前链表节点数是偶数
        return mid;
    }
    ListNode reverseNode(ListNode head){
        //反转单链表
        if(head==null||head.next==null){
            return head;
        }
        ListNode next=head.next;
        head.next=null;
        ListNode newHead=reverseNode(next);
        next.next=head;
        return newHead;
    }
    public void reorderList(ListNode head) {
        //常规做法，直接将这两个链表断开，成两个部分
        ListNode mid=getMidNode(head);
        ListNode postHead=reverseNode(mid);
        ListNode p1=head,p2=postHead;
        while(p2!=null){
            ListNode next1=p1.next;
            ListNode next2=p2.next;
            p1.next=p2;
            p2.next=next1;
            p1=next1;
            p2=next2;
        }
    }
~~~



##### 官方解答：数组映射

~~~java
 public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }
        List<ListNode> list = new ArrayList<ListNode>();
        ListNode node = head;
        while (node != null) {
            list.add(node);
            node = node.next;
        }
        int i = 0, j = list.size() - 1;
        while (i < j) {
            list.get(i).next = list.get(j);
            i++;
            if (i == j) {
                break;
            }
            list.get(j).next = list.get(i);
            j--;
        }
        list.get(i).next = null;
    }
~~~



#### [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)

难度中等1440收藏分享切换为英文接收动态反馈

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

 

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

**提示：**

- `1 <= n <= 104`

通过次数312,446

提交次数478,16

##### dp方法

直接使用基本的动态规划即可，关键是分析出子问题

~~~java
 public int numSquares(int n) {
        //简单动态规划
        int[] dp=new int[n+1];
        for(int i=1;i<=n;i++){
            int ans=Integer.MAX_VALUE;
            for(int j=1;j*j<=i;j++){
                ans=Math.min(ans,dp[i-j*j]+1);
            }
            dp[i]=ans;
        }
        return dp[n];
}
~~~



#### [1411. 给 N x 3 网格图涂色的方案数](https://leetcode.cn/problems/number-of-ways-to-paint-n-3-grid/)

难度困难106收藏分享切换为英文接收动态反馈

你有一个 `n x 3` 的网格图 `grid` ，你需要用 **红，黄，绿** 三种颜色之一给每一个格子上色，且确保相邻格子颜色不同（也就是有相同水平边或者垂直边的格子颜色不同）。

给你网格图的行数 `n` 。

请你返回给 `grid` 涂色的方案数。由于答案可能会非常大，请你返回答案对 `10^9 + 7` 取余的结果。

 

**示例 1：**

```
输入：n = 1
输出：12
解释：总共有 12 种可行的方法：
```

**示例 2：**

```
输入：n = 2
输出：54
```

**示例 3：**

```
输入：n = 3
输出：246
```

**示例 4：**

```
输入：n = 7
输出：106494
```

**示例 5：**

```
输入：n = 5000
输出：30228214
```

 

**提示：**

- `n == grid.length`
- `grid[i].length == 3`
- `1 <= n <= 5000`

通过次数8,909

提交次数15,931

##### 我的解法

题目本身不算太复杂，主要是处理的逻辑步骤比较多，基本思路就是$n$行的网格的具体数量只能由于$n-1$行网格的颜色分布相关。

~~~java
 public int numOfWays(int n) {
        int[][] oneCount=new int[][]{{1,2,1},{2,1,2},{3,1,2},
            {1,2,3},{2,1,3},{3,1,3},
            {1,3,1},{2,3,1},{3,2,1},
            {1,3,2},{2,3,2},{3,2,3}
            };
        long ans=12;
        long mod=1000000007;
        Map<int[],Long> preItemCountMap=
            Arrays.asList(oneCount).stream().collect(Collectors.toMap(Function.identity(),k->1L));
        for(int i=2;i<=n;i++){
            ans=0;
            Map<int[],Long> updateCountMap=new HashMap();
            for(int[] item:oneCount){
                long itemCount=0;
                for(Map.Entry<int[],Long> entry:preItemCountMap.entrySet()){
                    boolean matched=true;
                    for(int j=0;j<item.length;j++){
                        if(item[j]==entry.getKey()[j]){
                            matched=false;
                            break;
                        }
                    }
                    if(matched){
                        itemCount+=entry.getValue();
                    }
                }
                if(itemCount>0){
                    itemCount%=mod;
                    updateCountMap.put(item,itemCount);
                    ans+=itemCount;
                }
            }
            preItemCountMap=updateCountMap;
        }
       return (int)(ans%mod);
    }
~~~



##### 官方解答一

把颜色一维数组转换成了单独的数字

~~~java
 static final int MOD = 1000000007;

    public int numOfWays(int n) {
        // 预处理出所有满足条件的 type
        List<Integer> types = new ArrayList<Integer>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    if (i != j && j != k) {
                        // 只要相邻的颜色不相同就行
                        // 将其以十进制的形式存储
                        types.add(i * 9 + j * 3 + k);
                    }
                }
            }
        }
        int typeCnt = types.size();
        // 预处理出所有可以作为相邻行的 type 对
        int[][] related = new int[typeCnt][typeCnt];
        for (int i = 0; i < typeCnt; ++i) {
            // 得到 types[i] 三个位置的颜色
            int x1 = types.get(i) / 9, x2 = types.get(i) / 3 % 3, x3 = types.get(i) % 3;
            for (int j = 0; j < typeCnt; ++j) {
                // 得到 types[j] 三个位置的颜色
                int y1 = types.get(j) / 9, y2 = types.get(j) / 3 % 3, y3 = types.get(j) % 3;
                // 对应位置不同色，才能作为相邻的行
                if (x1 != y1 && x2 != y2 && x3 != y3) {
                    related[i][j] = 1;
                }
            }
        }
        // 递推数组
        int[][] f = new int[n + 1][typeCnt];
        // 边界情况，第一行可以使用任何 type
        for (int i = 0; i < typeCnt; ++i) {
            f[1][i] = 1;
        }
        for (int i = 2; i <= n; ++i) {
            for (int j = 0; j < typeCnt; ++j) {
                for (int k = 0; k < typeCnt; ++k) {
                    // f[i][j] 等于所有 f[i - 1][k] 的和
                    // 其中 k 和 j 可以作为相邻的行
                    if (related[k][j] != 0) {
                        f[i][j] += f[i - 1][k];
                        f[i][j] %= MOD;
                    }
                }
            }
        }
        // 最终所有的 f[n][...] 之和即为答案
        int ans = 0;
        for (int i = 0; i < typeCnt; ++i) {
            ans += f[n][i];
            ans %= MOD;
        }
        return ans;
    }
~~~



#### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

难度中等1064收藏分享切换为英文接收动态反馈

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

 

**示例 1：**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

**示例 2：**

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
```

**示例 3：**

```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。
```

 

**提示：**

- `1 <= text1.length, text2.length <= 1000`
- `text1` 和 `text2` 仅由小写英文字符组成。

通过次数257,811

提交次数399,135

##### 我的解法

简单动态规划，注意边界条件处理。

~~~java
 public int longestCommonSubsequence(String text1, String text2) {
        int m=text1.length(),n=text2.length();
        int[][] dp=new int[m][n];
        int i,j;
        dp[0][0]=text1.charAt(0)==text2.charAt(0)?1:0;
        for(i=0;i<m;i++){
            for(j=0;j<n;j++){
                if(i==0&&j==0){
                    continue;
                }
                if(i==0){
                    dp[i][j]=Math.max(text1.charAt(i)==text2.charAt(j)?1:0,dp[i][j-1]);
                    continue;
                }
                if(j==0){
                    dp[i][j]=Math.max(text1.charAt(i)==text2.charAt(j)?1:0,dp[i-1][j]);
                    continue;
                }
                int carry=text1.charAt(i)==text2.charAt(j)?1:0;
                dp[i][j]=Math.max(carry+dp[i-1][j-1],Math.max(dp[i-1][j],dp[i][j-1]));
            }
        }
        return dp[m-1][n-1];
    }
~~~



##### 官方解法一

巧用额外空余数组

~~~java
public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            char c1 = text1.charAt(i - 1);
            for (int j = 1; j <= n; j++) {
                char c2 = text2.charAt(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
~~~



#### [202. 快乐数](https://leetcode.cn/problems/happy-number/)

难度简单996收藏分享切换为英文接收动态反馈

编写一个算法来判断一个数 `n` 是不是快乐数。

**「快乐数」** 定义为：

- 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
- 然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。
- 如果这个过程 **结果为** 1，那么这个数就是快乐数。

如果 `n` 是 *快乐数* 就返回 `true` ；不是，则返回 `false` 。

 

**示例 1：**

```
输入：n = 19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

**示例 2：**

```
输入：n = 2
输出：false
```

 

**提示：**

- `1 <= n <= 231 - 1`

通过次数277,141

提交次数439,594

##### 常规解法

直接进行枚举操作，由快乐数定义可知，快乐数的值是有限的（通过累加各个数值位的平方和），最终的结果要么最终是一个快乐数或者陷入一个死循环。**(java中没有阶乘符号^)**

~~~java
 int getNext(int n){
        int next=0;
        while(n>0){
            int mod=n%10;
            next=next+mod*mod;
            n/=10;
        }
        return next;
    }
    public boolean isHappy(int n) {
        Set<Integer> seen=new HashSet();
        int next=getNext(n);
        while(!seen.contains(next)){
            if(next==1){
                return true;
            }
            seen.add(next);
            next=getNext(next);
        }
        return false;
    }
~~~



#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

难度简单646收藏分享切换为英文接收动态反馈

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

 

**示例 1：**

```
输入: n = 5, m = 3
输出: 3
```

**示例 2：**

```
输入: n = 10, m = 17
输出: 2
```

 

**限制：**

- `1 <= n <= 10^5`
- `1 <= m <= 10^6`

通过次数160,121

提交次数242,847

##### 递归调用

最终保留的数和数字的具体内容无关，只是返回最终的索引。考虑对于问题规模$n$，第一个删除的索引应该是$(m-1)\%n$，删除之后还剩下$n-1$的问题规模，$n$问题的规模相当于$n-1$是起始点右移了$m$位。

~~~java
 public int lastRemaining(int n, int m) {
        int index=0;
        for(int i=2;i<=n;i++){
           index=(m+index)%i;
        }
        return index;
    }
~~~



#### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

难度中等1214收藏分享切换为英文接收动态反馈

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

 

**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

 

**提示：**

- `1 <= s.length <= 16`
- `s` 仅由小写英文字母组成

通过次数216,393

提交次数295,807

##### 直接递归求解

~~~java
 private boolean isParlindrome(String s){
        if(s==null||s.length()==0){
            return false;
        }
        int left=0,right=s.length()-1;
        while(left<right){
            if(s.charAt(left)!=s.charAt(right)){
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    

    public void backtrace(List<List<String>> pathList,List<String> path,String s,int startIndex){
        //回溯方法和剪枝方法应用
        if(startIndex==s.length()){
            pathList.add(new ArrayList(path));
            return;
        }
        for(int i=1;i+startIndex<=s.length();i++){
            String sub=s.substring(startIndex,startIndex+i);
            if(isParlindrome(sub)){
                path.add(sub);
                backtrace(pathList,path,s,startIndex+i);
                path.remove(path.size()-1);
            }
        }
    }
    public List<List<String>> partition(String s) {
       List<List<String>> pathList=new ArrayList();
       backtrace(pathList,new ArrayList(),s,0);
       return pathList;
    }
~~~





##### 官方解答1：dfs和动态规划

~~~java
boolean[][] f;
    List<List<String>> ret = new ArrayList<List<String>>();
    List<String> ans = new ArrayList<String>();
    int n;

    public List<List<String>> partition(String s) {
        n = s.length();
        f = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(f[i], true);
        }

        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                f[i][j] = (s.charAt(i) == s.charAt(j)) && f[i + 1][j - 1];
            }
        }
        dfs(s, 0);
        return ret;
    }

    public void dfs(String s, int i) {
        if (i == n) {
            ret.add(new ArrayList<String>(ans));
            return;
        }
        for (int j = i; j < n; ++j) {
            if (f[i][j]) {
                ans.add(s.substring(i, j + 1));
                dfs(s, j + 1);
                ans.remove(ans.size() - 1);
            }
        }
    }
~~~



#### [133. 克隆图](https://leetcode.cn/problems/clone-graph/)

难度中等518收藏分享切换为英文接收动态反馈

给你无向 **[连通](https://baike.baidu.com/item/连通图/6460995?fr=aladdin)** 图中一个节点的引用，请你返回该图的 [**深拷贝**](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)（克隆）。

图中的每个节点都包含它的值 `val`（`int`） 和其邻居的列表（`list[Node]`）。

```
class Node {
    public int val;
    public List<Node> neighbors;
}
```

 

**测试用例格式：**

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（`val = 1`），第二个节点值为 2（`val = 2`），以此类推。该图在测试用例中使用邻接列表表示。

**邻接列表** 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 **给定节点的拷贝** 作为对克隆图的引用返回。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/133_clone_graph_question.png)

```
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/graph.png)

```
输入：adjList = [[]]
输出：[[]]
解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。
```

**示例 3：**

```
输入：adjList = []
输出：[]
解释：这个图是空的，它不含任何节点。
```

**示例 4：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/graph-1.png)

```
输入：adjList = [[2],[1]]
输出：[[2],[1]]
```

 

**提示：**

1. 节点数不超过 100 。
2. 每个节点值 `Node.val` 都是唯一的，`1 <= Node.val <= 100`。
3. 无向图是一个[简单图](https://baike.baidu.com/item/简单图/1680528?fr=aladdin)，这意味着图中没有重复的边，也没有自环。
4. 由于图是无向的，如果节点 *p* 是节点 *q* 的邻居，那么节点 *q* 也必须是节点 *p* 的邻居。
5. 图是连通图，你可以从给定节点访问到所有节点。

通过次数97,068

提交次数141,726



##### 直接解法

采用dfs构建方法，由于图中可能存在循环引用问题，所以需要实现缓存对应的数据节点。

~~~java
 Map<Integer,Node> map=new HashMap();
    Node dfs(Node initNode){
        if(initNode==null){
            return initNode;
        }
        if(map.containsKey(initNode.val)){
            return map.get(initNode.val);
        }
        Node cloneNode=new Node(initNode.val);
        map.put(initNode.val,cloneNode);
        List<Node> neighbors=initNode.neighbors;
        if(neighbors!=null){
            List<Node> cloneNeignbors=new ArrayList();
            for(Node node:neighbors){
                cloneNeignbors.add(dfs(node));
            }
            cloneNode.neighbors=cloneNeignbors;
        }
        return cloneNode;
    }
    public Node cloneGraph(Node node) {
        Node cloneNode=dfs(node);
        return cloneNode;
    }
~~~



##### 官方题解2：广度优先

~~~java
 public Node cloneGraph(Node node) {
        if (node == null) {
            return node;
        }

        HashMap<Node, Node> visited = new HashMap();

        // 将题目给定的节点添加到队列
        LinkedList<Node> queue = new LinkedList<Node> ();
        queue.add(node);
        // 克隆第一个节点并存储到哈希表中
        visited.put(node, new Node(node.val, new ArrayList()));

        // 广度优先搜索
        while (!queue.isEmpty()) {
            // 取出队列的头节点
            Node n = queue.remove();
            // 遍历该节点的邻居
            for (Node neighbor: n.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    // 如果没有被访问过，就克隆并存储在哈希表中
                    visited.put(neighbor, new Node(neighbor.val, new ArrayList()));
                    // 将邻居节点加入队列中
                    queue.add(neighbor);
                }
                // 更新当前节点的邻居列表
                visited.get(n).neighbors.add(visited.get(neighbor));
            }
        }

        return visited.get(node);
    }
~~~



#### [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

难度困难807收藏分享切换为英文接收动态反馈

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

 

**示例 1:**

```
输入: [7,5,6,4]
输出: 5
```

 

**限制：**

```
0 <= 数组长度 <= 50000
```

通过次数158,971

提交次数321,434

##### 二分解法

只管的暴力破解法时间复杂度是$O(n^2)$，会超时。考虑到采用分治算法。逆序队的数字来源分为三个部分，左边、右边、较大的数字在左边小的数字在右边，一共三种场景。主要是第三种场景，如何在$O(n)$的时间内计算迭代，考虑到是比较数值大小，可以结合归并排序。

**数组中的第一个元素存放的是逆序队个数**

~~~java
int[] combineSort(int[] nums,int left,int right){
        //归并排序，同时将归并排序的逆序对返回，作为数组首位
        if(left>right){
            return new int[]{0};
        }
        if(left==right){
            return new int[]{0,nums[left]};
        }
        int mid=(left+right)/2;
        int[] leftNums=combineSort(nums,left,mid);
        int[] rightNums=combineSort(nums,mid+1,right);
        int reverseCount=leftNums[0]+rightNums[0];
        int[] sortNums=new int[leftNums.length+rightNums.length-1];
        int i=1,j=1,k=1;
        while(i<leftNums.length&&j<rightNums.length){
            if(leftNums[i]<=rightNums[j]){
                sortNums[k++]=leftNums[i++];
            }else{
                //逆序数量添加
                reverseCount+=leftNums.length-i;
                sortNums[k++]=rightNums[j++];
            }
        }
        while(i<leftNums.length){
             sortNums[k++]=leftNums[i++];
        }
        while(j<rightNums.length){
             sortNums[k++]=rightNums[j++];
        }
        sortNums[0]=reverseCount;
        return sortNums;
    }
    public int reversePairs(int[] nums) {
         return combineSort(nums,0,nums.length-1)[0];
    }
~~~













## 2022年8月

#### [456. *132 模式](https://leetcode.cn/problems/132-pattern/)

难度中等682收藏分享切换为英文接收动态反馈

给你一个整数数组 `nums` ，数组中共有 `n` 个整数。**132 模式的子序列** 由三个整数 `nums[i]`、`nums[j]` 和 `nums[k]` 组成，并同时满足：`i < j < k` 和 `nums[i] < nums[k] < nums[j]` 。

如果 `nums` 中存在 **132 模式的子序列** ，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：nums = [1,2,3,4]
输出：false
解释：序列中不存在 132 模式的子序列。
```

**示例 2：**

```
输入：nums = [3,1,4,2]
输出：true
解释：序列中有 1 个 132 模式的子序列： [1, 4, 2] 。
```

**示例 3：**

```
输入：nums = [-1,3,2,0]
输出：true
解释：序列中有 3 个 132 模式的的子序列：[-1, 3, 2]、[-1, 3, 0] 和 [-1, 2, 0] 。
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 2 * 105`
- `-109 <= nums[i] <= 109`

通过次数68,683

提交次数188,480

##### 解法一

由题目第一次想到的是通过单调栈来解决这个问题，以中间坐标$j$所在的数作为基准，由题目所知，第一次查找$j$左边比$nums[j]$小的最小的数值$i$，采用单调递增栈；下一步查找对应的$k$,数值$nums[k]$应该是比$nums[j]$小的最大的值，但是无法在$O(n)$时间复杂度查找出来。退而求其次，采用排序map方式，通过查找$nums[i]+1$的上边界，然后进行比较。

~~~java
 public boolean find132pattern(int[] nums) {
        //以中间数值索引j作为基础，向左右两边查找扩展
        //左边找到比坐标j索引小的最小的值，右边找到比坐标索引j小的最大的一个值
        Deque<Integer> lowLeft=new LinkedList();
        if(nums.length<3){
            return false;
        }
        //用来查找j的右边界
        TreeMap<Integer,Integer> rightMap=new TreeMap();
        for(int num:nums){
            rightMap.put(num,rightMap.getOrDefault(num,0)+1);
        }
        int i;
        //找到左边界，左边界是单调递增栈
        for(i=0;i<nums.length;i++){
            while(!lowLeft.isEmpty()&&nums[lowLeft.peek()]>=nums[i]){
                lowLeft.pop();
            }
            int lowIndex=lowLeft.isEmpty()?-1:lowLeft.getLast();
            Integer count=rightMap.get(nums[i]);
            count--;
            if(count==0){
                rightMap.remove(nums[i]);
            }else{
                rightMap.put(nums[i],count);
            }
            if(lowIndex!=-1){
                //查找比当前值的最接近的上界
                Integer nextKey=rightMap.ceilingKey(nums[lowIndex]+1);
                if(nextKey!=null&&nextKey<nums[i]){
                    return true;
                }
            }
            lowLeft.push(i);
        }
      
        return false;
    }
~~~



##### 官方解法2：单调栈

看不太明白，单调栈里面维护的是中间大的那个坐标$k$。

~~~java
 public boolean find132pattern(int[] nums) {
        int n = nums.length;
        Deque<Integer> candidateK = new LinkedList<Integer>();
        candidateK.push(nums[n - 1]);
        int maxK = Integer.MIN_VALUE;

        for (int i = n - 2; i >= 0; --i) {
            if (nums[i] < maxK) {
                return true;
            }
            while (!candidateK.isEmpty() && nums[i] > candidateK.peek()) {
                maxK = candidateK.pop();
            }
            if (nums[i] > maxK) {
                candidateK.push(nums[i]);
            }
        }

        return false;
    }

~~~



#### [815. 公交路线](https://leetcode.cn/problems/bus-routes/)

难度困难300收藏分享切换为英文接收动态反馈

给你一个数组 `routes` ，表示一系列公交线路，其中每个 `routes[i]` 表示一条公交线路，第 `i` 辆公交车将会在上面循环行驶。

- 例如，路线 `routes[0] = [1, 5, 7]` 表示第 `0` 辆公交车会一直按序列 `1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ...` 这样的车站路线行驶。

现在从 `source` 车站出发（初始时不在公交车上），要前往 `target` 车站。 期间仅可乘坐公交车。

求出 **最少乘坐的公交车数量** 。如果不可能到达终点车站，返回 `-1` 。

 

**示例 1：**

```
输入：routes = [[1,2,7],[3,6,7]], source = 1, target = 6
输出：2
解释：最优策略是先乘坐第一辆公交车到达车站 7 , 然后换乘第二辆公交车到车站 6 。 
```

**示例 2：**

```
输入：routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12
输出：-1
```

 

**提示：**

- `1 <= routes.length <= 500`.
- `1 <= routes[i].length <= 105`
- `routes[i]` 中的所有值 **互不相同**
- `sum(routes[i].length) <= 105`
- `0 <= routes[i][j] < 106`
- `0 <= source, target < 106`

通过次数32,021

提交次数73,204

##### 我的解法

建立车站和公交路线的map映射，然后采用bfs遍历。本质上是建立公交路线的图结构，官方采用的是邻接数组方式

~~~java
public int numBusesToDestination(int[][] routes, int source, int target) {
        //bfs算法，首先建立公交车站和路线索引的映射关系，然后通过bfs遍历路径
        Map<Integer,Set<Integer>> mp=new HashMap();
        Set<Integer> seen=new HashSet();
        int i,j;
        for(i=0;i<routes.length;i++){
            for(j=0;j<routes[i].length;j++){
                mp.computeIfAbsent(routes[i][j],k->new HashSet()).add(i);
            }
        }
    	//注意边界
        if(source==target){
            return 0;
        }
        Queue<Integer> queue=new LinkedList();
        queue.addAll(mp.getOrDefault(source,new HashSet()));
        int steps=0;
        while(!queue.isEmpty()){
            int size=queue.size();
            steps++;
            for(i=0;i<size;i++){
                int routeIndex=queue.poll();
                if(seen.contains(routeIndex)){
                    continue;
                }
                seen.add(routeIndex);
                Set<Integer> buses=Arrays.stream(routes[routeIndex]).boxed().collect(Collectors.toSet());       if(buses.contains(target)){
                    return steps;
                }
                for(Integer bus:buses){
                    queue.addAll(mp.getOrDefault(bus,new HashSet()));
                }
            }
        }
        return -1;
    }
~~~



#### [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

难度中等1252收藏分享切换为英文接收动态反馈

**[Trie](https://baike.baidu.com/item/字典树/9825209?fr=aladdin)**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

 

**示例：**

```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

 

**提示：**

- `1 <= word.length, prefix.length <= 2000`
- `word` 和 `prefix` 仅由小写英文字母组成
- `insert`、`search` 和 `startsWith` 调用次数 **总计** 不超过 `3 * 104` 次

通过次数215,941

提交次数300,034

##### 二叉搜索树思想

比较简单

~~~java
 Tree root;
    public Trie() {

    }
    
    public void insert(String word) {
        if(root==null){
            root=new Tree(word);
            return;
        }
        Tree endTree=searchPos(null,root,word);
        int cmp=endTree.word.compareTo(word);
        if(cmp>0){
            endTree.left=new Tree(word);
        }else if(cmp<0){
            endTree.right=new Tree(word);
        }
    }
    private Tree searchPos(Tree parent,Tree tree,String word){
        if(tree==null){
            return parent;
        }
        int cmp=tree.word.compareTo(word);
        if(cmp==0){
            return tree;
        }else if(cmp>0){
            return searchPos(tree,tree.left,word);
        }else{
            return searchPos(tree,tree.right,word);
        }
    }
    
    public boolean search(String word) {
       return searchWord(root,word);
    }
    private boolean searchWord(Tree root,String word){
        if(root==null){
            return false;
        }
        int cmp=root.word.compareTo(word);
        if(cmp==0){
            return true;
        }else if(cmp>0){
            return searchWord(root.left,word);
        }else{
            return searchWord(root.right,word);
        }
    }
    
    boolean startsWith(Tree root,String prefix){
          if(root==null){
            return false;
        }
        if(root.word.startsWith(prefix)){
            return true;
        }
        int cmp=root.word.compareTo(prefix);
        if(cmp>0){
            return startsWith(root.left,prefix);
        }else{
            return startsWith(root.right,prefix);
        }
    }
    public boolean startsWith(String prefix) {
      return startsWith(root,prefix);
    }
    static class Tree{
        String word;
        public Tree(String word){
            this.word=word;
        }
        Tree left;
        Tree right;
    }
~~~



##### 官方解答：单个字符

不是很容易理解，每个前缀树建立26个子节点，表示以为字母$a,b,c...z$开头的字符串，每一个字符占据树的一层

~~~java
 private Trie[] children;
    private boolean isEnd;

    public Trie() {
        children = new Trie[26];
        isEnd = false;
    }
    
    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int index = ch - 'a';
            if (node.children[index] == null) {
                node.children[index] = new Trie();
            }
            node = node.children[index];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        Trie node = searchPrefix(word);
        return node != null && node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }

    private Trie searchPrefix(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            int index = ch - 'a';
            if (node.children[index] == null) {
                return null;
            }
            node = node.children[index];
        }
        return node;
    }
~~~



#### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

难度困难1933收藏分享切换为英文接收动态反馈

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

 

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

 

**提示：**

- `0 <= s.length <= 3 * 104`
- `s[i]` 为 `'('` 或 `')'`

通过次数296,189

提交次数805,372

##### 我的解法

第一选择是通过递归来实现，不过超时了。采用动态规划方式，定义

$dp[i]$表示以$i$坐标结尾的最长括号，显然只有$i$坐标是$)$该值才可能不为空。查看子问题：

- 如果$i-1$的字符是$($，那么最长长度是$2+dp[i-2]$
- 如果$i-1$的字符是$)$，那么查看以$i-1$结尾的最长括号的左边界的前一个数值是否是$($

~~~java
public int longestValidParentheses(String s) {
        //dp算法实现，定义dp[i]表示第i个字符结尾的字符串的最大
        //括号长度
        int[] dp=new int[s.length()+1];
        int ans=0;
        char[] arr=s.toCharArray();
        for(int i=0;i<arr.length;i++){
            char ch=arr[i];
            if(ch=='('){
                continue;
            }
            //当前结尾的字符是)，如果上一个字符是(,那么结果是s[i]=s[i-2]+2;
            //否则判断与当前字符匹配的位置的字符是否是(
            int j=i-1;
            if(i-1>=0){
                if(arr[i-1]=='('){
                    dp[i+1]=dp[i-1]+2;
                }else{
                    //查看与)对应的上一个字符
                    j=i-dp[i]-1;
                    if(j>=0&&arr[j]=='('){
                        dp[i+1]=dp[j]+i-j+1;
                    }
                }

            }
            ans=Math.max(ans,dp[i+1]);
        }
        return ans;
    }
~~~



##### 官方解法二：栈

栈底存放的是最后一个没有匹配的$)$的下标，每次遍历的时候，遇到$($入栈，遇到$)$出栈，并更新括号最大长度。初始化添加-1表示未匹配的

$)$下标。

~~~java
 public int longestValidParentheses(String s) {
        int maxans = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    maxans = Math.max(maxans, i - stack.peek());
                }
            }
        }
        return maxans;
    }
~~~



##### 官方解法三：不使用额外空间

利用括号的性质，遍历的时候记录左括号和右括号的数量，从左到右遍历的时候，如果右括号数量大于左括号数量，那么充值；如果相等，那么更新最长有效括号。注意对于$(()$这种场景，是无法获取的，这种情况采用逆序方式从右边到左遍历。

~~~java
 public int longestValidParentheses(String s) {
        int left = 0, right = 0, maxlength = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = Math.max(maxlength, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = Math.max(maxlength, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }
        return maxlength;
    }
~~~



#### [10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)

难度困难3128

给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

- `'.'` 匹配任意单个字符
- `'*'` 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 **整个** 字符串 `s`的，而不是部分字符串。

**示例 1：**

```
输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**示例 3：**

```
输入：s = "ab", p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

 

**提示：**

- `1 <= s.length <= 20`
- `1 <= p.length <= 30`
- `s` 只包含从 `a-z` 的小写字母。
- `p` 只包含从 `a-z` 的小写字母，以及字符 `.` 和 `*`。
- 保证每次出现字符 `*` 时，前面都匹配到有效的字符

通过次数300,836

提交次数949,197

##### 递归方式匹配

难点主要在于对于$*$符号的处理

~~~java
 boolean regexMatch(char[] arr1,int i,char[] arr2,int j){
        if(i==arr1.length||j==arr2.length){
            //还有此种匹配a和ab*这种
            if(j!=arr2.length){
                if(j+1<arr2.length && arr2[j+1]=='*'){
                    return regexMatch(arr1,i,arr2,j+2);
                }
                return false;
            }
            return i==arr1.length && j==arr2.length;
        }
        //正常匹配
        if(j==arr2.length-1||(j+1<arr2.length && arr2[j+1]!='*')){
            if(arr1[i]==arr2[j]||arr2[j]=='.'){
                return regexMatch(arr1,i+1,arr2,j+1);
            }else{
                return false;
            }
        }
        //通配符*匹配，*跳过
        if(arr1[i]!=arr2[j]&&arr2[j]!='.'){
            return regexMatch(arr1,i,arr2,j+2);
        }
        return regexMatch(arr1,i,arr2,j+2)||regexMatch(arr1,i+1,arr2,j)||regexMatch(arr1,i+1,arr2,j+2);
    }
    public boolean isMatch(String s, String p) {
        char[] arr1=s.toCharArray();
        char[] arr2=p.toCharArray();
        return regexMatch(arr1,0,arr2,0);
    }
~~~



##### 官方解法二：动态规划

对于\*有两种匹配原则，字符\*需要和其它字符组合使用

1. 匹配 $s$ 末尾的一个字符，将该字符扔掉，而该组合还可以继续进行匹配。 $f[i-1][j]$
2. 不匹配字符，将该组合扔掉，不再进行匹配。$f[i][j-2]$

~~~java
public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }
~~~



#### [6. Z 字形变换](https://leetcode.cn/problems/zigzag-conversion/)

难度中等1753

将一个给定字符串 `s` 根据给定的行数 `numRows` ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 `"PAYPALISHIRING"` 行数为 `3` 时，排列如下：

```
P   A   H   N
A P L S I I G
Y   I   R
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"PAHNAPLSIIGYIR"`。

请你实现这个将字符串进行指定行数变换的函数：

```
string convert(string s, int numRows);
```

 

**示例 1：**

```
输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
```

**示例 2：**

```
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I
```

**示例 3：**

```
输入：s = "A", numRows = 1
输出："A"
```

 

**提示：**

- `1 <= s.length <= 1000`
- `s` 由英文字母（小写和大写）、`','` 和 `'.'` 组成
- `1 <= numRows <= 1000`

通过次数465,322

提交次数894,832

##### 基本解法

主要是对于周期的处理

~~~java
  public String convert(String s, int numRows) {
        if(numRows==1||numRows>=s.length()){
            return s;
        }
        //单个周期需要的元素个数
        int r=numRows*2-2;
        //最大的列数，可能达不到
        int col=(s.length()+r-1)/r*(numRows-1);
        StringBuilder[] ansList=new StringBuilder[numRows];
        for(int i=0;i<ansList.length;i++){
            ansList[i]=new StringBuilder();
        }
        //当前字符对应的行数
        int x=-1;
        for(int i=0;i<s.length();i++){
            int mod=i%r;
            if(mod<=numRows-1){
                x=mod;
            }else{
                x--;
            }
            ansList[x].append(s.charAt(i));
        }
        StringBuilder ans=new StringBuilder();
        for(StringBuilder row:ansList){
            ans.append(row);
        }
        return ans.toString();
    }
~~~



#### [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

难度中等752

给定一个 `*m* x *n*` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法**。**



 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

```
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)

```
输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[0].length`
- `1 <= m, n <= 200`
- `-231 <= matrix[i][j] <= 231 - 1`

 

**进阶：**

- 一个直观的解决方案是使用  `O(*m**n*)` 的额外空间，但这并不是一个好的解决方案。
- 一个简单的改进方案是使用 `O(*m* + *n*)` 的额外空间，但这仍然不是最好的解决方案。
- 你能想出一个仅使用常量空间的解决方案吗？

##### 直接解法

~~~java
 public void setZeroes(int[][] matrix) {
        Set<Integer> zeroRowSet=new HashSet();
        Set<Integer> zeroColSet=new HashSet();
        int i,j;
        for(i=0;i<matrix.length;i++){
            for(j=0;j<matrix[i].length;j++){
                if(matrix[i][j]==0){
                    zeroRowSet.add(i);
                    zeroColSet.add(j);
                }
            }
        }
        //原地赋值，行
        for(Integer rowIndex:zeroRowSet){
            int[] rowMatrix=matrix[rowIndex];
            Arrays.fill(rowMatrix,0);
        }
        //原地赋值，列
        for(Integer colIndex:zeroColSet){
            for(i=0;i<matrix.length;i++){
                matrix[i][colIndex]=0;
            }
        }

    }
~~~



##### 解法二：使用两个标记变量

以第一行和第一列作为标记，指示当前行或者列是否需要赋0。对于$matrix[i][j]$，只要$matrix[0][j]==0或者matrix[i][0]==0$，那么这个数值就赋值为0，初始化时候应该查找第一行和第一列的标记。

~~~java
 public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean flagCol0 = false, flagRow0 = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                flagCol0 = true;
            }
        }
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                flagRow0 = true;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (flagCol0) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        if (flagRow0) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
    }
~~~



##### 解法三：使用一个标记变量

~~~java
public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean flagCol0 = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                flagCol0 = true;
            }
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
            if (flagCol0) {
                matrix[i][0] = 0;
            }
        }
    }
~~~



#### [223. 矩形面积](https://leetcode.cn/problems/rectangle-area/)

难度中等195

给你 **二维** 平面上两个 **由直线构成且边与坐标轴平行/垂直** 的矩形，请你计算并返回两个矩形覆盖的总面积。

每个矩形由其 **左下** 顶点和 **右上** 顶点坐标表示：

- 第一个矩形由其左下顶点 `(ax1, ay1)` 和右上顶点 `(ax2, ay2)` 定义。
- 第二个矩形由其左下顶点 `(bx1, by1)` 和右上顶点 `(bx2, by2)` 定义。

 

**示例 1：**

![Rectangle Area](https://assets.leetcode.com/uploads/2021/05/08/rectangle-plane.png)

```
输入：ax1 = -3, ay1 = 0, ax2 = 3, ay2 = 4, bx1 = 0, by1 = -1, bx2 = 9, by2 = 2
输出：45
```

**示例 2：**

```
输入：ax1 = -2, ay1 = -2, ax2 = 2, ay2 = 2, bx1 = -2, by1 = -2, bx2 = 2, by2 = 2
输出：16
```

 

**提示：**

- `-104 <= ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 <= 104`

通过次数49,953

提交次数94,065



##### 常规解法

矩阵相交，矩阵的左右或者上下坐标就是将这些坐标进行排序，取得中间两位作为边界。

~~~java
 /**
     *计算重叠的面积
     */
    int computeOverlapArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2){
          //判断是否有交叉面积
        if(ay1>=by2||ay2<=by1||ax2<=bx1||ax1>=bx2){
            return 0;
        }
        int[] row=new int[]{ax1,ax2,bx1,bx2};
        Arrays.sort(row);
        int[] col=new int[]{ay1,ay2,by1,by2};
        Arrays.sort(col);
        return (col[2]-col[1])*(row[2]-row[1]);
    }
    public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        int sumArea=(ay2-ay1)*(ax2-ax1)+(by2-by1)*(bx2-bx1);
        int overlapArea=computeOverlapArea(ax1,ay1,ax2,ay2,bx1,by1,bx2,by2);
        return sumArea-overlapArea;
    }
~~~



##### 官方解答

~~~java
 public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        int area1 = (ax2 - ax1) * (ay2 - ay1), area2 = (bx2 - bx1) * (by2 - by1);
        int overlapWidth = Math.min(ax2, bx2) - Math.max(ax1, bx1), overlapHeight = Math.min(ay2, by2) - Math.max(ay1, by1);
        int overlapArea = Math.max(overlapWidth, 0) * Math.max(overlapHeight, 0);
        return area1 + area2 - overlapArea;
    }
~~~





#### [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

难度中等1879

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

```
输入：n = 3
输出：5
```

**示例 2：**

```
输入：n = 1
输出：1
```

 

**提示：**

- `1 <= n <= 19`

通过次数278,408

提交次数394,423

##### 我的解法：动态规划

二叉搜索树的数量和具体的数值没有关系，只和数值的数量有关，定义$dp[n]$表示$n$个数值不同二叉搜索树数量，那么子问题是以为数值$1,2,3...n$为根节点的二叉搜索树之和。
$$
dp[n]=\sum_{i=1}^{n}dp[i-1]*dp[n-i]
$$


~~~java
 public int numTrees(int n) {
        //动态规划，递推公式为dp[n]=dp[i-1]*dp[n-i] (i从1到n)
        int[] dp=new int[n+1];
        dp[0]=1;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=i;j++){
                dp[i]+=dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }
~~~



#### [119. 杨辉三角 II](https://leetcode.cn/problems/pascals-triangle-ii/)

难度简单414

给定一个非负索引 `rowIndex`，返回「杨辉三角」的第 `rowIndex` 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

![img](https://pic.leetcode-cn.com/1626927345-DZmfxB-PascalTriangleAnimated2.gif)

 

**示例 1:**

```
输入: rowIndex = 3
输出: [1,3,3,1]
```

**示例 2:**

```
输入: rowIndex = 0
输出: [1]
```

**示例 3:**

```
输入: rowIndex = 1
输出: [1,1]
```

 

**提示:**

- `0 <= rowIndex <= 33`

 

**进阶：**

你可以优化你的算法到 `*O*(*rowIndex*)` 空间复杂度吗？

通过次数217,818

提交次数317,325

##### 基本解法

~~~java
 public List<Integer> getRow(int rowIndex) {
        int[] preRow=new int[]{1};
        int[] nextRow=preRow;
        int i=0;
        while(i<rowIndex){
            nextRow=new int[preRow.length+1];
            for(int j=0;j<=(nextRow.length-1)/2;j++){
                if(j==0){
                    nextRow[j]=1;
                    nextRow[nextRow.length-1]=1;
                    continue;
                }
                nextRow[j]=preRow[j-1]+preRow[j];
                nextRow[nextRow.length-1-j]=nextRow[j];
            }
            preRow=nextRow;
            i++;
        }
        return Arrays.stream(nextRow).boxed().collect(Collectors.toList());
    }
~~~

##### 官方解法

~~~java
 public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            row.add(0);
            for (int j = i; j > 0; --j) {
                row.set(j, row.get(j) + row.get(j - 1));
            }
        }
        return row;
    }
~~~



#### [44. 通配符匹配](https://leetcode.cn/problems/wildcard-matching/)

难度困难924

给定一个字符串 (`s`) 和一个字符模式 (`p`) ，实现一个支持 `'?'` 和 `'*'` 的通配符匹配。

```
'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
```

两个字符串**完全匹配**才算匹配成功。

**说明:**

- `s` 可能为空，且只包含从 `a-z` 的小写字母。
- `p` 可能为空，且只包含从 `a-z` 的小写字母，以及字符 `?` 和 `*`。

**示例 1:**

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入:
s = "aa"
p = "*"
输出: true
解释: '*' 可以匹配任意字符串。
```

**示例 3:**

```
输入:
s = "cb"
p = "?a"
输出: false
解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
```

**示例 4:**

```
输入:
s = "adceb"
p = "*a*b"
输出: true
解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
```

**示例 5:**

```
输入:
s = "acdcb"
p = "a*c?b"
输出: false
```

通过次数114,972

提交次数343,043

##### 失败尝试：递归

这种解法会导致超时

~~~java
	//递归方法调用会超时
    boolean regexMatch(char[] arr1,int i,char[] arr2,int j){
        if(i==arr1.length||j==arr2.length){
            if(j!=arr2.length){
                if(arr2[j]=='*'){
                    return regexMatch(arr1,i,arr2,j+1);
                }
            }
            return i==arr1.length&&j==arr2.length;
        }
        char a1=arr1[i];
        char a2=arr2[j];
        if(a2=='*'){
            //分别表示跳过，多次匹配，只匹配一次
            return regexMatch(arr1,i,arr2,j+1)||regexMatch(arr1,i+1,arr2,j)||
                regexMatch(arr1,i+1,arr2,j+1);
        }
        if(a1==a2||a2=='?'){
            return regexMatch(arr1,i+1,arr2,j+1);
        }
        return false;
    }
~~~



##### 我的解法：dp

实现逻辑步骤比较繁琐，没有使用额外的空余边界，导致逻辑处理比较麻烦。

~~~java
 public boolean isMatch(String s, String p) {
        boolean[][] dp=new boolean[p.length()][s.length()];
        if(s.length()==0||p.length()==0){
           if(s.length()==0){
               // "***"和""匹配场景
               boolean allDot=true;
               for(char ch:p.toCharArray()){
                   if(ch!='*'){
                       allDot=false;
                   }
               }
               return allDot;
           }
           return s.length()==0&&p.length()==0;
        }
        int i,j;
        for(i=0;i<p.length();i++){
            char cp=p.charAt(i);
            for(j=0;j<s.length();j++){
                char cs=s.charAt(j);
                boolean matched=cs==cp||cp=='?'||cp=='*';
                //边界条件
                if(i==0||j==0){
                    if(i==0&&j==0){
                        dp[i][j]=matched;
                    }else if(i==0){
                        dp[i][j]=cp=='*'&&dp[i][j-1];
                    }else{
                        //j==0情况下
                        if(cp=='*'){
                            if(dp[i-1][j]){
                                dp[i][j]=true;
                            }
                        }
                        //*a和a的匹配情景,也可能会出现?*a和a的匹配场景
                        if(!dp[i-1][j]||!matched){
                            dp[i][j]=false;
                        }else{
                            //matched的情况下，检查前面i-1个字符是否都是*
                            boolean allAste=true;
                            for(int k=0;k<=i-1;k++){
                                if(p.charAt(k)!='*'){
                                    allAste=false;
                                    break;
                                }
                            }
                            if(allAste){
                                dp[i][j]=true;
                            }
                        }
                    }
                }else{
                    if(cp=='*'){
                        if(dp[i][j-1]||dp[i-1][j]){
                            dp[i][j]=true;
                        }
                    }
                    if(dp[i-1][j-1]&&matched){
                        dp[i][j]=true;
                    }
                }
            }
        }
        return dp[p.length()-1][s.length()-1];
    }
~~~



##### 解法二：官方题解

对于*的匹配子问题$dp[i-1][j]$其实内部已经包含了$dp[i-1][j-1]$。

~~~ 
public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

~~~



#### [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

难度简单633

给定两个字符串 `*s*` 和 `*t*` ，编写一个函数来判断 `*t*` 是否是 `*s*` 的字母异位词。

**注意：**若 `*s*` 和 `*t*` 中每个字符出现的次数都相同，则称 `*s*` 和 `*t*` 互为字母异位词。

 

**示例 1:**

```
输入: s = "anagram", t = "nagaram"
输出: true
```

**示例 2:**

```
输入: s = "rat", t = "car"
输出: false
```

 

**提示:**

- `1 <= s.length, t.length <= 5 * 104`
- `s` 和 `t` 仅包含小写字母

 

**进阶:** 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

通过次数464,905

提交次数709,094

##### 直接解法

~~~java
 public boolean isAnagram(String s, String t) {
        if(s.length()!=t.length()){
            return false;
        }
        int[] count=new int[26];
        for(char ch:s.toCharArray()){
            count[ch-'a']++;
        }
        for(char ch:t.toCharArray()){
            count[ch-'a']--;
        }
        for(int cnt:count){
            if(cnt!=0){
                return false;
            }
        }
        return true;
    }
~~~



#### [260. 只出现一次的数字 III](https://leetcode.cn/problems/single-number-iii/)

难度中等636

给定一个整数数组 `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 **任意顺序** 返回答案。

 

**进阶：**你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？

 

**示例 1：**

```
输入：nums = [1,2,1,3,2,5]
输出：[3,5]
解释：[5, 3] 也是有效的答案。
```

**示例 2：**

```
输入：nums = [-1,0]
输出：[-1,0]
```

**示例 3：**

```
输入：nums = [0,1]
输出：[1,0]
```

**提示：**

- `2 <= nums.length <= 3 * 104`
- `-231 <= nums[i] <= 231 - 1`
- 除两个只出现一次的整数外，`nums` 中的其他数字都出现两次

通过次数96,214

提交次数131,159

##### 集合方式

~~~java
public int[] singleNumber(int[] nums) {
        Set<Integer> visited=new HashSet();
        for(int num:nums){
            if(visited.contains(num)){
                visited.remove(num);
            }else{
                visited.add(num);
            }
        }
        return visited.stream().mapToInt(Integer::intValue).toArray();
    }
~~~



##### 解法二：异或运算

~~~java
 public int[] singleNumber(int[] nums) {
        int xorsum = 0;
        for (int num : nums) {
            xorsum ^= num;
        }
        // 防止溢出
        int lsb = (xorsum == Integer.MIN_VALUE ? xorsum : xorsum & (-xorsum));
        int type1 = 0, type2 = 0;
        for (int num : nums) {
            if ((num & lsb) != 0) {
                type1 ^= num;
            } else {
                type2 ^= num;
            }
        }
        return new int[]{type1, type2};
    }
~~~



#### [58. 最后一个单词的长度](https://leetcode.cn/problems/length-of-last-word/)

难度简单480

给你一个字符串 `s`，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 **最后一个** 单词的长度。

**单词** 是指仅由字母组成、不包含任何空格字符的最大子字符串。

 

**示例 1：**

```
输入：s = "Hello World"
输出：5
解释：最后一个单词是“World”，长度为5。
```

**示例 2：**

```
输入：s = "   fly me   to   the moon  "
输出：4
解释：最后一个单词是“moon”，长度为4。
```

**示例 3：**

```
输入：s = "luffy is still joyboy"
输出：6
解释：最后一个单词是长度为6的“joyboy”。
```

 

**提示：**

- `1 <= s.length <= 104`
- `s` 仅有英文字母和空格 `' '` 组成
- `s` 中至少存在一个单词

通过次数344,880

提交次数843,728

###### 基本解法

~~~java
 public int lengthOfLastWord(String s) {
        int left ,right=s.length()-1;
        while(s.charAt(right)==' '){
            right--;
        }
        left=right;
        while(left-1>=0&&s.charAt(left-1)!=' '){
            left--;
        }
        return right-left+1;
    }
~~~



#### [77. 组合](https://leetcode.cn/problems/combinations/)

难度中等1072

给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案。

 

**示例 1：**

```
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

**示例 2：**

```
输入：n = 1, k = 1
输出：[[1]]
```

 

**提示：**

- `1 <= n <= 20`
- `1 <= k <= n`

通过次数397,193

提交次数514,596

##### 常规递归解法

~~~java
 List<List<Integer>> ansList=new ArrayList();
    void combination(int start,int n,int k,List<Integer> ans){
        if(ans.size()==k){
            ansList.add(new ArrayList(ans));
            return;
        }
        //减枝，剩余的数据凑不够k个数
        if(n-start+1+ans.size()<k){
            return;
        }
        //选择start
        ans.add(start);
        combination(start+1,n,k,ans);
        ans.remove(ans.size()-1);
        //不选择start
        combination(start+1,n,k,ans);
    }
    public List<List<Integer>> combine(int n, int k) {
        combination(1,n,k,new ArrayList());
        return ansList;
    }
~~~



#### [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

难度中等897

给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。

 

**示例 1：**

```
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

**示例 2：**

```
输入：nums = [0]
输出：[[],[0]]
```

 

**提示：**

- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`

通过次数227,645

提交次数357,976

~~~java
	List<List<Integer>> ansList=new ArrayList();
    void subset(int[] nums,int startIndex,List<Integer> ans,int k){
        if(ans.size()==k){
            ansList.add(new ArrayList(ans));
            return;
        }
        int remain=k-ans.size();
        if(remain>nums.length-startIndex){
            return;
        }
        for(int i=startIndex;i<nums.length;i++){
            if(i>startIndex&&nums[i]==nums[i-1]){
                continue;
            }
            ans.add(nums[i]);
            subset(nums,i+1,ans,k);
            ans.remove(ans.size()-1);
        }
    }
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        ansList.add(new ArrayList());
        for(int i=1;i<=nums.length;i++){
            subset(nums,0,new ArrayList(),i);
        }
        return ansList;
    }
~~~



#### [86. 分隔链表](https://leetcode.cn/problems/partition-list/)

难度中等604

给你一个链表的头节点 `head` 和一个特定值 `x` ，请你对链表进行分隔，使得所有 **小于** `x` 的节点都出现在 **大于或等于** `x` 的节点之前。

你应当 **保留** 两个分区中每个节点的初始相对位置。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)

```
输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]
```

**示例 2：**

```
输入：head = [2,1], x = 2
输出：[1,2]
```

 

**提示：**

- 链表中节点的数目在范围 `[0, 200]` 内
- `-100 <= Node.val <= 100`
- `-200 <= x <= 200`

通过次数166,547

提交次数262,116

##### 直接遍历

~~~java
public ListNode partition(ListNode head, int x) {
        if(head==null||head.next==null){
            return head;
        }
        ListNode lt=new ListNode();
        ListNode d1=new ListNode();
        d1.next=head;
        ListNode p1=d1;
        ListNode p2=lt;
        while(p1.next!=null){
            if(p1.next.val<x){
                p2.next=p1.next;
                p1.next=p1.next.next;
                p2=p2.next;
                p2.next=null;
                continue;
            }
            p1=p1.next;
        }
        p2.next=d1.next;
        return lt.next;
    }
~~~



#### [30. 串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)

难度困难791

给定一个字符串 `s` 和一些 **长度相同** 的单词 `words` **。**找出 `s` 中恰好可以由 `words` 中所有单词串联形成的子串的起始位置。

注意子串要与 `words` 中的单词完全匹配，**中间不能有其他字符** ，但不需要考虑 `words` 中单词串联的顺序。

 

**示例 1：**

```
输入：s = "barfoothefoobarman", words = ["foo","bar"]
输出：[0,9]
解释：
从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。
```

**示例 2：**

```
输入：s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
输出：[]
```

**示例 3：**

```
输入：s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
输出：[6,9,12]
```

 

**提示：**

- `1 <= s.length <= 104`
- `s` 由小写英文字母组成
- `1 <= words.length <= 5000`
- `1 <= words[i].length <= 30`
- `words[i]` 由小写英文字母组成

通过次数133,862

提交次数340,594

##### 递归剪枝优化

~~~java
 int[] count(String[] words){
        int[] arr=new int[26];
        for(String word:words){
            for(int i=0;i<word.length();i++){
                arr[word.charAt(i)-'a']++;
            }
        }
        return arr;
    }
    boolean equalArr(int[] arr1,int[] arr2){
        if(arr1.length!=arr2.length){
            return false;
        }
        for(int i=0;i<arr1.length;i++){
            if(arr1[i]!=arr2[i]){
                return false;
            }
        }
        return true;
    }
    boolean containsAllWord(String s,int startIndex,Map<String,Integer> countMap,int wl){
        if(startIndex==s.length()){
            return true;
        }
        String substr=s.substring(startIndex,startIndex+wl);
        if(!countMap.containsKey(substr)||countMap.get(substr)==0){
            return false;
        }
        countMap.put(substr,countMap.get(substr)-1);
        boolean containsAll=containsAllWord(s,startIndex+wl,countMap,wl);
        countMap.put(substr,countMap.get(substr)+1);
        return containsAll;
    }
    public List<Integer> findSubstring(String s, String[] words) {
        int wl=words[0].length();
        int totalLength=wl*words.length;
        int[] wlArr=count(words);
        Map<String,Integer> countMap=new HashMap();
        for(String word:words){
            countMap.put(word,countMap.getOrDefault(word,0)+1);
        }
        List<Integer> ans=new ArrayList();
        for(int i=0;i<s.length()-totalLength+1;i++){
            String substr=s.substring(i,i+totalLength);
            int[] subArr=count(new String[]{substr});
            if(!equalArr(wlArr,subArr)){
                continue;
            }
            if(containsAllWord(substr,0,countMap,wl)){
                ans.add(i);
            }
        }
        return ans;
    }
~~~



#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

难度简单589

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 `appendTail` 和 `deleteHead` ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，`deleteHead` 操作返回 -1 )

 

**示例 1：**

```
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
```

**示例 2：**

```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```

**提示：**

- `1 <= values <= 10000`
- `最多会对 appendTail、deleteHead 进行 10000 次调用`

通过次数455,344

提交次数643,511

##### 我的答案

比较简单，栈和队列的区别就是栈只能单向操作，并且和队列的出入顺序相反，根据这个特性可知，栈如果出栈和入站两次的话，那么就和队列的顺序一致了。

~~~java
    
	//s1就是两次入栈的数值，顺序和队列相同
	Stack<Integer> s1=new Stack();
    Stack<Integer> s2=new Stack();
    public CQueue() {

    }
    
    public void appendTail(int value) {
        s1.push(value);
    }
    
    public int deleteHead() {
        if(s2.isEmpty()){
            while(!s1.isEmpty()){
                s2.push(s1.pop());
            }
        }
        return s2.isEmpty()?-1:s2.pop();
    }
~~~



#### [52. N皇后 II](https://leetcode.cn/problems/n-queens-ii/)

难度困难381

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n × n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回 **n 皇后问题** 不同的解决方案的数量。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/13/queens.jpg)

```
输入：n = 4
输出：2
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

**示例 2：**

```
输入：n = 1
输出：1
```

 

**提示：**

- `1 <= n <= 9`

通过次数99,520

提交次数121,034

##### 我的解法

n皇后问题主要是回溯和减枝方法的结合，皇后位置冲突有三种方式：

- 行冲突：每一行只能有一个皇后，遍历的时候可以手动控制
- 列冲突：和已有的皇后位置进行比较，排除掉列冲突的
- 对角冲突：两个皇后左边斜率是45或者135

~~~java
int backtrace(boolean[][] visited,int lineIndex,List<int[]> path){
        if(lineIndex==visited.length){
            return 1;
        }
        int ans=0;
        for(int j=0;j<visited[lineIndex].length;j++){
            boolean conflict=false;
            for(int[] item:path){
                //检查列冲突
                if(item[1]==j){
                    conflict=true;
                    break;
                }
                //检查对角冲突
                if(lineIndex+j==item[0]+item[1]||lineIndex+item[1]==item[0]+j){
                    conflict=true;
                    break;
                }
            }
            if(conflict){
                continue;
            }
            visited[lineIndex][j]=true;
            path.add(new int[]{lineIndex,j});
            ans+=backtrace(visited,lineIndex+1,path);
            visited[lineIndex][j]=false;
            path.remove(path.size()-1);
        }
        return ans;
    }
    public int totalNQueens(int n) {
        boolean[][] visited=new boolean[n][n];
        return backtrace(visited,0,new ArrayList());
    }
~~~



#### [59. 螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)

难度中等774

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/13/spiraln.jpg)

```
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]
```

**示例 2：**

```
输入：n = 1
输出：[[1]]
```

 

**提示：**

- `1 <= n <= 20`

通过次数226,191

提交次数298,561

##### 常规解法

螺旋矩阵的遍历顺序是固定的，依次为左->右，上->下，右->左，下->上。遍历过程中记住遍历的位置，一旦到达边界或者到达重复位置，那么调整方向。

~~~java
public int[][] generateMatrix(int n) {
        int[][] ans=new int[n][n];
        int i=0,j=-1;
        int[][] directions={{0,1},{1,0},{0,-1},{-1,0}};
        boolean[][] visited=new boolean[n][n];
        int index=0;
        int k=1;
        while(k<=n*n){
            int[] dir=directions[index];
            int x=i+dir[0];
            int y=j+dir[1];
            while(k<=n*n&& x>=0&&x<n&&y>=0&&y<n&&!visited[x][y]){
                ans[x][y]=k;
                visited[x][y]=true;
                i=x;
                j=y;
                k++;
                x+=dir[0];
                y+=dir[1];
            }
            index=(index+1)%directions.length;
        }
        return ans;
    }
~~~



##### 官方题解

~~~java
 public int[][] generateMatrix(int n) {
        int maxNum = n * n;
        int curNum = 1;
        int[][] matrix = new int[n][n];
        int row = 0, column = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 右下左上
        int directionIndex = 0;
        while (curNum <= maxNum) {
            matrix[row][column] = curNum;
            curNum++;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= n || nextColumn < 0 || nextColumn >= n || matrix[nextRow][nextColumn] != 0) {
                directionIndex = (directionIndex + 1) % 4; // 顺时针旋转至下一个方向
            }
            row = row + directions[directionIndex][0];
            column = column + directions[directionIndex][1];
        }
        return matrix;
}        
~~~



#### [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)

难度中等715

给你一个有序数组 `nums` ，请你**[ 原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使得出现次数超过两次的元素**只出现两次** ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

 

**说明：**

为什么返回数值是整数，但输出的答案是数组呢？

请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

 

**示例 1：**

```
输入：nums = [1,1,1,2,2,3]
输出：5, nums = [1,1,2,2,3]
解释：函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。 不需要考虑数组中超出新长度后面的元素。
```

**示例 2：**

```
输入：nums = [0,0,1,1,1,1,2,3,3]
输出：7, nums = [0,0,1,1,2,3,3]
解释：函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。 不需要考虑数组中超出新长度后面的元素。
```

 

**提示：**

- `1 <= nums.length <= 3 * 104`
- `-104 <= nums[i] <= 104`
- `nums` 已按升序排列

通过次数194,469

提交次数313,678

##### 双指针

第一次碰到数值时，直接添加。比如数组$1,2,2,2,3,3,3$，碰到坐标$1$对应的$2$或者碰到坐标$4$对应的$3$时，直接添加。

~~~java
public int removeDuplicates(int[] nums) {
        //双指针
        int slow=0,fast=0;
        int count=0;
        for(;fast<nums.length;){
            if(fast==0){
                count++;
                slow=fast;
                fast++;
                continue;
            }
            while(fast<nums.length&&nums[fast]==nums[fast-1]){
                fast++;
                count++;
            }
            if(count==1){
                nums[++slow]=nums[fast];
                fast++;
            }else{
                nums[++slow]=nums[fast-1];
                count=1;
            }
        }
        return slow+1;
    }
~~~



##### 官方题解

~~~java
public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 2) {
            return n;
        }
        int slow = 2, fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }
~~~



#### [91. 解码方法](https://leetcode.cn/problems/decode-ways/)

难度中等1223

一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

```
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"11106"` 可以映射为：

- `"AAJF"` ，将消息分组为 `(1 1 10 6)`
- `"KJF"` ，将消息分组为 `(11 10 6)`

注意，消息不能分组为 `(1 11 06)` ，因为 `"06"` 不能映射为 `"F"` ，这是由于 `"6"` 和 `"06"` 在映射中并不等价。

给你一个只含数字的 **非空** 字符串 `s` ，请计算并返回 **解码** 方法的 **总数** 。

题目数据保证答案肯定是一个 **32 位** 的整数。

 

**示例 1：**

```
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
```

**示例 2：**

```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

**示例 3：**

```
输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。
含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
```

 

**提示：**

- `1 <= s.length <= 100`
- `s` 只包含数字，并且可能包含前导零。

通过次数230,712

提交次数709,663

##### 简单dp

基本dp算法实现，注意边界条件处理。

~~~java
 public int numDecodings(String s) {
        int[] dp=new int[s.length()+1];
        for(int i=1;i<dp.length;i++){
            char ch=s.charAt(i-1);
            if(ch>='1'&&ch<='9'){
                dp[i]+=(i==1?1:dp[i-1]);
            }
            if(i>=2){
                char preCh=s.charAt(i-2);
                int val=(preCh-'0')*10+ch-'0';
                if(val>=10&&val<=26){
                    dp[i]+=(i-2==0?1:dp[i-2]);
                }
            }
        }
        return dp[s.length()];
    }
~~~



#### [205. 同构字符串](https://leetcode.cn/problems/isomorphic-strings/)

难度简单501

给定两个字符串 `s` 和 `t` ，判断它们是否是同构的。

如果 `s` 中的字符可以按某种映射关系替换得到 `t` ，那么这两个字符串是同构的。

每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。

 

**示例 1:**

```
输入：s = "egg", t = "add"
输出：true
```

**示例 2：**

```
输入：s = "foo", t = "bar"
输出：false
```

**示例 3：**

```
输入：s = "paper", t = "title"
输出：true
```

 

**提示：**



- `1 <= s.length <= 5 * 104`
- `t.length == s.length`
- `s` 和 `t` 由任意有效的 ASCII 字符组成

通过次数161,028

提交次数323,077

##### 基本解法

注意条件中的映射关系，由于字符都是ASCII码，可以使用int数组来做映射，由于题目要求字符映射必须是一一映射，为了避免多个字符映射到同一个字符，额外创建了一个数组，用来标记哪些字符已经被映射

~~~java
public boolean isIsomorphic(String s, String t) {
        int[] mp=new int[129];
        //标记对应val是否已经被映射
        boolean[] visited=new boolean[129];
        if(s.length()!=t.length()){
            return false;
        }
        for(int i=0;i<s.length();i++){
            char c1=s.charAt(i);
            char c2=t.charAt(i);
           	//c1没有映射，那么确保c1映射的c2没有跟以前的字符匹配，比如字符ac和bb。
            if(mp[c1]==0){
                if(visited[c2+1]){
                    return false;
                }
                mp[c1]=c2+1;
                visited[c2+1]=true;
                continue;
            }
            if(mp[c1]!=c2+1){
                return false;
            }
        }
        return true;
    }
~~~



##### 官方：双向map

~~~java
public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> s2t = new HashMap<Character, Character>();
        Map<Character, Character> t2s = new HashMap<Character, Character>();
        int len = s.length();
        for (int i = 0; i < len; ++i) {
            char x = s.charAt(i), y = t.charAt(i);
            if ((s2t.containsKey(x) && s2t.get(x) != y) || (t2s.containsKey(y) && t2s.get(y) != x)) {
                return false;
            }
            s2t.put(x, y);
            t2s.put(y, x);
        }
        return true;
    }
~~~



#### [159. 至多包含两个不同字符的最长子串](https://leetcode.cn/problems/longest-substring-with-at-most-two-distinct-characters/)

难度中等180

给定一个字符串 ***s\*** ，找出 **至多** 包含两个不同字符的最长子串 ***t\*** ，并返回该子串的长度。

**示例 1:**

```
输入: "eceba"
输出: 3
解释: t 是 "ece"，长度为3。
```

**示例 2:**

```
输入: "ccaabbb"
输出: 5
解释: t 是 "aabbb"，长度为5。
```

通过次数25,286

提交次数45,342

##### 我的解法

基本思路就是采用滑动窗口，维护一个最长的字符串在滑动窗口中，遍历过程中，移除左边界字符，然后更新最长字符串。难点在于更新字符串时候的边界处理。

~~~java
 public int lengthOfLongestSubstringTwoDistinct(String s) {
        int ans=0;
        int count=0;
        if(s==null||s.length()==0){
            return 0;
        }
        int i=0;
        int start=0;
        int[] mp=new int[128];
        while(start<s.length()){
            while(count<=2&&i<s.length()){
                char ch=s.charAt(i);
                mp[ch]++;
                if(mp[ch]==1){
                    count++;
                }
                i++;
            }
            if(i==s.length()){
                if(count<=2){
                    ans=Math.max(ans,i-start);
                }else{
                    ans=Math.max(ans,i-1-start);
                }
                break;
            }
            ans=Math.max(ans,i-1-start);
            if(--mp[s.charAt(start)]==0){
                count--;
            }     
            start++;   
        }
        return ans;
    }
~~~



##### 官方解答

~~~java
public int lengthOfLongestSubstringTwoDistinct(String s) {
    int n = s.length();
    if (n < 3) return n;

    // sliding window left and right pointers
    int left = 0;
    int right = 0;
    // hashmap character -> its rightmost position 
    // in the sliding window
    HashMap<Character, Integer> hashmap = new HashMap<Character, Integer>();

    int max_len = 2;

    while (right < n) {
      // slidewindow contains less than 3 characters
      if (hashmap.size() < 3)
        hashmap.put(s.charAt(right), right++);

      // slidewindow contains 3 characters
      if (hashmap.size() == 3) {
        // delete the leftmost character
        int del_idx = Collections.min(hashmap.values());
        hashmap.remove(s.charAt(del_idx));
        // move left pointer of the slidewindow
        left = del_idx + 1;
      }

      max_len = Math.max(max_len, right - left);
    }
    return max_len;
  }
~~~



##### 网上优秀解法

~~~~java
char[] s;
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        this.s = s.toCharArray();
        return lengthOfLongestSubstringTwoDistinct();
    }
    private int lengthOfLongestSubstringTwoDistinct() {
        int n = s.length;
        if (n <= 1) {
            return n;
        }
        int[] map = new int[128];
        int diffs = 0;
        int L = 0, R = 0;
        int maxLen = 0;
        while (R < n) {
            if (map[s[R]] == 0) {
                diffs++;
            }
            map[s[R++]]++;
            if (diffs <= 2) {
                maxLen = Math.max(maxLen, R - L);
            } else {
                while (map[s[L]] > 1) {
                    map[s[L++]]--;
                }
                map[s[L++]]--;
                diffs--;
            }
        }
        return maxLen;
    }
}
~~~~



##### 精简解法

每次循环过程中，拓展左右边界。

~~~java
public int lengthOfLongestSubstringTwoDistinct(String s) {
        int ans=0;
        int count=0;
        if(s==null||s.length()==0){
            return 0;
        }
        int i=0;
        int start=0;
        int[] mp=new int[128];
        while(i<s.length()){
            if(count<=2){
                char ch=s.charAt(i);
                if(mp[ch]==0){
                    count++;
                }
                mp[ch]++;
            }
            if(count<=2){
                ans=Math.max(ans,i-start+1);
                i++;
            }else{
                //拓展左边界
                char rmCh=s.charAt(start);
                mp[rmCh]--;
                if(mp[rmCh]==0){
                    count--;
                }
                start++;
                //拓展右边界
                if(count<=2){
                    i++;
                }
            }
        }
        return ans;
    }
~~~



#### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

难度中等1439

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

 

**示例 1：**

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

**示例 2：**

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

 

**提示：**

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`

通过次数307,551

提交次数590,894



##### 典型背包问题

题目要求将数组进行二分，判断二分之后的数组和是否相等等价于从数组中查找数组和等于当前数组和一半的值是否存在。由题可知是一个典型背包问题，背包问题转移方程如下：（哪个作为一维数组都一样）**($i-1$表示不重复)**
$$
dp[i][w]=max(nums[i]+dp[i-1][w-weight[i]],dp[i-1][w])  //前提是weight[i]<=w
$$
如果以权重作为一维数组，那么方程式
$$
dp[w][i]=max(dp[w][i-1],dp[w-weight[i]][i-1]+nums[i])
$$


~~~java
 int bag(int[] nums,int sum){
        int[][] dp=new int[sum+1][nums.length+1];
        int i,j;
        for(i=0;i<=sum;i++){
            for(j=0;j<=nums.length;j++){
                if(i==0||j==0){
                    dp[i][j]=0;
                    continue;
                }
                int num=nums[j-1];
                if(num>i){
                    dp[i][j]=dp[i][j-1];
                }else{
                    dp[i][j]=Math.max(dp[i][j-1],num+dp[i-num][j-1]);
                }
            }
        }
        return dp[sum][nums.length];
    }
    public boolean canPartition(int[] nums) {
        //背包问题
        int sum=Arrays.stream(nums).sum();
        if(sum!=sum/2*2){
            return false;
        }
        Arrays.sort(nums);
        int target=sum/2;
        int maxWeight=bag(nums,target);
        return target==maxWeight;
    }
~~~



#### [219. 存在重复元素 II](https://leetcode.cn/problems/contains-duplicate-ii/)

难度简单503

给你一个整数数组 `nums` 和一个整数 `k` ，判断数组中是否存在两个 **不同的索引** `i` 和 `j` ，满足 `nums[i] == nums[j]` 且 `abs(i - j) <= k` 。如果存在，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：nums = [1,2,3,1], k = 3
输出：true
```

**示例 2：**

```
输入：nums = [1,0,1,1], k = 1
输出：true
```

**示例 3：**

```
输入：nums = [1,2,3,1,2,3], k = 2
输出：false
```

 

 

**提示：**

- `1 <= nums.length <= 105`
- `-109 <= nums[i] <= 109`
- `0 <= k <= 105`

通过次数194,517

提交次数438,263

##### 滑动窗口

维护一个窗口宽度为$k+1$的集合，然后顺序遍历，超过窗口的话，就将左边界移除。

~~~java
 public boolean containsNearbyDuplicate(int[] nums, int k) {
        if(nums.length<2||k==0){
            return false;
        }
        //k的窗口
        Set<Integer> st=new HashSet();
        int i=0;
        while(i<nums.length&&i<=k){
            if(st.contains(nums[i])){
                return true;
            }
            st.add(nums[i]);
            i++;
        }
        while(i<nums.length){
            st.remove(nums[i-k-1]);
            if(st.contains(nums[i])){
                return true;
            }
            st.add(nums[i]);
            i++;
        }
        return false;
    }
~~~



##### 官方解法：map

~~~java
public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int length = nums.length;
        for (int i = 0; i < length; i++) {
            int num = nums[i];
            if (map.containsKey(num) && i - map.get(num) <= k) {
                return true;
            }
            map.put(num, i);
        }
        return false;
}
~~~



#### [220. 存在重复元素 III](https://leetcode.cn/problems/contains-duplicate-iii/)

难度中等639

给你一个整数数组 `nums` 和两个整数 `k` 和 `t` 。请你判断是否存在 **两个不同下标** `i` 和 `j`，使得 `abs(nums[i] - nums[j]) <= t` ，同时又满足 `abs(i - j) <= k` 。

如果存在则返回 `true`，不存在返回 `false`。

 

**示例 1：**

```
输入：nums = [1,2,3,1], k = 3, t = 0
输出：true
```

**示例 2：**

```
输入：nums = [1,0,1,1], k = 1, t = 2
输出：true
```

**示例 3：**

```
输入：nums = [1,5,9,1,5,9], k = 2, t = 3
输出：false
```

 

**提示：**

- `0 <= nums.length <= 2 * 104`
- `-231 <= nums[i] <= 231 - 1`
- `0 <= k <= 104`
- `0 <= t <= 231 - 1`

通过次数84,010

提交次数288,640



##### 边界查找

使用$TreeMap$来查找最接近的值，这种查找时间复杂度是$log(n)$，**注意整型数组越界问题。**

~~~java
  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if(k==0||nums.length<2){
            return false;
        }
        TreeSet<Integer> ts=new TreeSet();
        int i=0;
        while(i<nums.length){
            if(i-k-1>=0){
                ts.remove(nums[i-k-1]);
            }
            Integer next=ts.ceiling(nums[i]);
            //注意整型越界问题
            if(next!=null&& next-nums[i]>=0&&next-nums[i]<=t){
                return true;
            }
            next=ts.floor(nums[i]);
            if(next!=null&&nums[i]-next>=0&&nums[i]-next<=t){
                return true;
            }
            ts.add(nums[i]);
            i++;
        }
        return false;
    }
~~~



##### 官方题解2：

为了避免整型溢出，将数值转换成了$long$类型

~~~java
 public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int n = nums.length;
        TreeSet<Long> set = new TreeSet<Long>();
        for (int i = 0; i < n; i++) {
            Long ceiling = set.ceiling((long) nums[i] - (long) t);
            if (ceiling != null && ceiling <= (long) nums[i] + (long) t) {
                return true;
            }
            set.add((long) nums[i]);
            //上面的语句已经包含了n-k下标的数值，可以删除
            if (i >= k) {
                set.remove((long) nums[i - k]);
            }
        }
        return false;
    }
~~~



##### 官方解法三：桶排序

保持桶的大小为$t+1$，比较滑动窗口范围内的数值是否落在相同的桶内或者相邻的桶内。

注意桶的划分方法，当数值为负数时候，比如桶的宽度是10，划分区间是(-10,-1),(0,9),(10,20)，所以负数需要额外添加1，确保-10和-1落在一个桶里。

~~~java
 public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int n = nums.length;
        Map<Long, Long> map = new HashMap<Long, Long>();
        long w = (long) t + 1;
        for (int i = 0; i < n; i++) {
            long id = getID(nums[i], w);
            if (map.containsKey(id)) {
                return true;
            }
            if (map.containsKey(id - 1) && Math.abs(nums[i] - map.get(id - 1)) < w) {
                return true;
            }
            if (map.containsKey(id + 1) && Math.abs(nums[i] - map.get(id + 1)) < w) {
                return true;
            }
            map.put(id, (long) nums[i]);
            if (i >= k) {
                map.remove(getID(nums[i - k], w));
            }
        }
        return false;
    }

    public long getID(long x, long w) {
        if (x >= 0) {
            return x / w;
        }
        return (x + 1) / w - 1;
    }
~~~



#### [97. 交错字符串](https://leetcode.cn/problems/interleaving-string/)

难度中等744

给定三个字符串 `s1`、`s2`、`s3`，请你帮忙验证 `s3` 是否是由 `s1` 和 `s2` **交错** 组成的。

两个字符串 `s` 和 `t` **交错** 的定义与过程如下，其中每个字符串都会被分割成若干 **非空** 子字符串：

- `s = s1 + s2 + ... + sn`
- `t = t1 + t2 + ... + tm`
- `|n - m| <= 1`
- **交错** 是 `s1 + t1 + s2 + t2 + s3 + t3 + ...` 或者 `t1 + s1 + t2 + s2 + t3 + s3 + ...`

**注意：**`a + b` 意味着字符串 `a` 和 `b` 连接。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)

```
输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出：true
```

**示例 2：**

```
输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
输出：false
```

**示例 3：**

```
输入：s1 = "", s2 = "", s3 = ""
输出：true
```

 

**提示：**

- `0 <= s1.length, s2.length <= 100`
- `0 <= s3.length <= 200`
- `s1`、`s2`、和 `s3` 都由小写英文字母组成

 

**进阶：**您能否仅使用 `O(s2.length)` 额外的内存空间来解决它?

通过次数93,577

提交次数207,605

##### 我的解法：dp

比较容易想到dp算法，做了部分减枝。

~~~java
 int[] count(String[] stArr){
        int[] mp=new int[26];
        for(String st:stArr){
            for(char ch:st.toCharArray()){
                mp[ch-'a']++;
            }
        }
        return mp;
    }
    public boolean isInterleave(String s1, String s2, String s3) {
        if(s1.length()+s2.length()!=s3.length()){
            return false;
        }
        //减枝
        int[] combineMp=count(new String[]{s1,s2});
        int[] s3Mp=count(new String[]{s3});
        int i,j;
        for(i=0;i<s3Mp.length;i++){
            if(s3Mp[i]!=combineMp[i]){
                return false;
            }
        }
        boolean[][] dp=new boolean[s1.length()+1][s2.length()+1];
        dp[0][0]=true;
        for(i=1;i<=s1.length();i++){
            char ch=s1.charAt(i-1);
            dp[i][0]=dp[i-1][0]&&ch==s3.charAt(i-1);
        }
        for(j=1;j<=s2.length();j++){
            char ch=s2.charAt(j-1);
            dp[0][j]=dp[0][j-1]&&ch==s3.charAt(j-1);
        }
        for(i=1;i<=s1.length();i++){
            char c1=s1.charAt(i-1);
            for(j=1;j<=s2.length();j++){
                char c2=s2.charAt(j-1);
                char c3=s3.charAt(i+j-1);
                dp[i][j]=(c1==c3&&dp[i-1][j])||(c2==c3&&dp[i][j-1]);
            }
        }
        return dp[s1.length()][s2.length()];
    }
~~~



##### 官方解法二：滚动数组优化

~~~java
 public boolean isInterleave(String s1, String s2, String s3) {
        int n = s1.length(), m = s2.length(), t = s3.length();

        if (n + m != t) {
            return false;
        }

        boolean[] f = new boolean[m + 1];

        f[0] = true;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                int p = i + j - 1;
                if (i > 0) {
                    f[j] = f[j] && s1.charAt(i - 1) == s3.charAt(p);
                }
                if (j > 0) {
                    f[j] = f[j] || (f[j - 1] && s2.charAt(j - 1) == s3.charAt(p));
                }
            }
        }

        return f[m];
    }
~~~



#### [60. 排列序列](https://leetcode.cn/problems/permutation-sequence/)

难度困难695

给出集合 `[1,2,3,...,n]`，其所有元素共有 `n!` 种排列。

按大小顺序列出所有排列情况，并一一标记，当 `n = 3` 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 `n` 和 `k`，返回第 `k` 个排列。

 

**示例 1：**

```
输入：n = 3, k = 3
输出："213"
```

**示例 2：**

```
输入：n = 4, k = 9
输出："2314"
```

**示例 3：**

```
输入：n = 3, k = 1
输出："123"
```

 

**提示：**

- `1 <= n <= 9`
- `1 <= k <= n!`

通过次数112,629

提交次数211,653

##### 直接求解

跟之前查找全排列的下一个数值类似。

~~~java
  void swapInt(int[] nums,int left,int right){
        int tmp=nums[left];
        nums[left]=nums[right];
        nums[right]=tmp;
    }
    void reverse(int[] nums,int left,int right){
        while(left<right){
            swapInt(nums,left,right);
            left++;
            right--;
        }
    }
    void nextLarge(int[] nums){
        //查找当前数值的下一个大的数
        int i=nums.length-1,j;
        while(i-1>=0&&nums[i-1]>=nums[i]){
            i--;
        }
        if(i>=1){
            j=nums.length-1;
            while(nums[j]<=nums[i-1]){
                j--;
            }
            swapInt(nums,i-1,j);
        }
        reverse(nums,i,nums.length-1);
    }
    public String getPermutation(int n, int k) {
        int[] nums=new int[n];
        for(int i=1;i<=n;i++){
            nums[i-1]=i;
        }
        k--;
        while(k>0){
            nextLarge(nums);
            k--;
        }
        StringBuilder ans=new StringBuilder();
        for(int num:nums){
            ans.append(num);
        }
        return ans.toString();
    }
~~~



##### 官方解答：缩放

基于以下的结论

- 对于n个不同的元素（例如1,2,3...n），它们可以组成的排列总数目是$n!$

按照全排列的首个元素$a1$为例

- a1为1的个数是$(n-1)!$
- a1为2的个数是$(n-1)!$
- a1为n的个数是$(n-1)!$

那么由此可以得到首个元素和$k$的关系
$$
a_{1}=\lfloor\frac{k-1}{(n-1)!}\rfloor+1
$$


~~~java
public String getPermutation(int n, int k) {
        int[] factorial = new int[n];
        factorial[0] = 1;
        for (int i = 1; i < n; ++i) {
            factorial[i] = factorial[i - 1] * i;
        }

        --k;
        StringBuffer ans = new StringBuffer();
        int[] valid = new int[n + 1];
        Arrays.fill(valid, 1);
        for (int i = 1; i <= n; ++i) {
            int order = k / factorial[n - i] + 1;
            //order对应该序列位的数值顺序
            for (int j = 1; j <= n; ++j) {
                order -= valid[j];
                if (order == 0) {
                    ans.append(j);
                    valid[j] = 0;
                    break;
                }
            }
            k %= factorial[n - i];
        }
        return ans.toString();
    }
~~~



#### [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)

难度简单835

给定一个已排序的链表的头 `head` ， *删除所有重复的元素，使每个元素只出现一次* 。返回 *已排序的链表* 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/01/04/list1.jpg)

```
输入：head = [1,1,2]
输出：[1,2]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/01/04/list2.jpg)

```
输入：head = [1,1,2,3,3]
输出：[1,2,3]
```

 

**提示：**

- 链表中节点数目在范围 `[0, 300]` 内
- `-100 <= Node.val <= 100`
- 题目数据保证链表已经按升序 **排列**

通过次数468,515

提交次数875,992

##### 直接求解

~~~java
public ListNode deleteDuplicates(ListNode head) {
        ListNode p=head;
        if(p==null){
            return p;
        }
        while(p.next!=null){
            if(p.val==p.next.val){
                ListNode tmp=p.next;
                p.next=p.next.next;
                tmp=null;
                continue;
            }
            p=p.next;
        }
        return head;
    }
~~~



#### [89. 格雷编码](https://leetcode.cn/problems/gray-code/)

难度中等528

**n 位格雷码序列** 是一个由 `2n` 个整数组成的序列，其中：

- 每个整数都在范围 `[0, 2n - 1]` 内（含 `0` 和 `2n - 1`）
- 第一个整数是 `0`
- 一个整数在序列中出现 **不超过一次**
- 每对 **相邻** 整数的二进制表示 **恰好一位不同** ，且
- **第一个** 和 **最后一个** 整数的二进制表示 **恰好一位不同**

给你一个整数 `n` ，返回任一有效的 **n 位格雷码序列** 。

 

**示例 1：**

```
输入：n = 2
输出：[0,1,3,2]
解释：
[0,1,3,2] 的二进制表示是 [00,01,11,10] 。
- 00 和 01 有一位不同
- 01 和 11 有一位不同
- 11 和 10 有一位不同
- 10 和 00 有一位不同
[0,2,3,1] 也是一个有效的格雷码序列，其二进制表示是 [00,10,11,01] 。
- 00 和 10 有一位不同
- 10 和 11 有一位不同
- 11 和 01 有一位不同
- 01 和 00 有一位不同
```

**示例 2：**

```
输入：n = 1
输出：[0,1]
```

 

**提示：**

- `1 <= n <= 16`

通过次数100,861

提交次数134,817

##### 观察数值

通过观察有效的格雷编码查找规律，比如以下，$n$可以由于前面$n-1$个格雷码逆序构成

~~~
n=1
0,1
n=2
11,10
n=3
110,111,101,100
~~~

~~~java
public List<Integer> grayCode(int n) {
        //尝试画出常见的n=3的格雷编码，观察规律
        //n层的值可以由于1-n-1层的值逆向推出
        List<Integer> ans=new ArrayList();
        ans.add(0);
        ans.add(1);
        int k=1;
        int base=1;
        while(++k<=n){
            base*=2;
            int levelSize=ans.size();
            for(int i=levelSize-1;i>=0;i--){
                int num=base+ans.get(i);
                ans.add(num);
            }
        }
        return ans;
    }
~~~



#### [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

难度中等695

编写一个高效的算法来判断 `m x n` 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/mat2.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
输出：false
```

 

**提示：**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 100`
- `-104 <= matrix[i][j], target <= 104`

通过次数257,907

提交次数537,922

##### 二分查找

题目描述可知，将二维矩阵按照行展开就是一个升序的数组，所以可以直接用二分查找

~~~java
public boolean searchMatrix(int[][] matrix, int target) {
        //二分搜索，将二维矩阵搜索转换成一维矩阵
        int m=matrix.length;
        int n=matrix[0].length;
        int low=0,high=m*n-1;
        while(low<=high){
            int mid=(high+low)/2;
            int x=mid/n;
            int y=mid%n;
            if(matrix[x][y]==target){
                return true;
            }else if(matrix[x][y]>target){
                high=mid-1;
            }else{
                low=mid+1;
            }
        }
        return false;
    }
~~~

















# SQL 查询

## [178. 分数排名](https://leetcode.cn/problems/rank-scores/)

难度中等955收藏分享切换为英文接收动态反馈

SQL架构

表: `Scores`

```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| score       | decimal |
+-------------+---------+
Id是该表的主键。
该表的每一行都包含了一场比赛的分数。Score是一个有两位小数点的浮点值。
```

 

编写 SQL 查询对分数进行排序。排名按以下规则计算:

- 分数应按从高到低排列。
- 如果两个分数相等，那么两个分数的排名应该相同。
- 在排名相同的分数后，排名数应该是下一个连续的整数。换句话说，排名之间不应该有空缺的数字。

按 `score` 降序返回结果表。

查询结果格式如下所示。

 

**示例 1:**

```
输入: 
Scores 表:
+----+-------+
| id | score |
+----+-------+
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |
+----+-------+
输出: 
+-------+------+
| score | rank |
+-------+------+
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |
+-------+------+
```

通过次数160,284

提交次数264,057

### 解法一：

~~~sql
select a.Score as score,
    (select count(distinct b.Score) from Scores as b where b.Score>=a.Score) as `rank`
from Scores as a
order by a.Score DESC
~~~





## [181. 超过经理收入的员工](https://leetcode.cn/problems/employees-earning-more-than-their-managers/)

难度简单523收藏分享切换为英文接收动态反馈

SQL架构

表：`Employee` 

```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
| salary      | int     |
| managerId   | int     |
+-------------+---------+
Id是该表的主键。
该表的每一行都表示雇员的ID、姓名、工资和经理的ID。
```

 

编写一个SQL查询来查找收入比经理高的员工。

以 **任意顺序** 返回结果表。

查询结果格式如下所示。

 

**示例 1:**

```
输入: 
Employee 表:
+----+-------+--------+-----------+
| id | name  | salary | managerId |
+----+-------+--------+-----------+
| 1  | Joe   | 70000  | 3         |
| 2  | Henry | 80000  | 4         |
| 3  | Sam   | 60000  | Null      |
| 4  | Max   | 90000  | Null      |
+----+-------+--------+-----------+
输出: 
+----------+
| Employee |
+----------+
| Joe      |
+----------+
解释: Joe 是唯一挣得比经理多的雇员。
```

通过次数222,053

提交次数321,074

### 解法一

~~~sql
select a.name `Employee` 
from Employee a ,Employee b
where a.managerId=b.id and a.salary>b.salary;
~~~



### 解法二

~~~java
SELECT
     a.NAME AS Employee
FROM Employee AS a JOIN Employee AS b
     ON a.ManagerId = b.Id
     AND a.Salary > b.Salary
~~~



## [176. 第二高的薪水](https://leetcode.cn/problems/second-highest-salary/)

难度中等1104收藏分享切换为英文接收动态反馈

SQL架构

`Employee` 表：

```
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
id 是这个表的主键。
表的每一行包含员工的工资信息。
```

 

编写一个 SQL 查询，获取并返回 `Employee` 表中第二高的薪水 。如果不存在第二高的薪水，查询应该返回 `null` 。

查询结果如下例所示。

 

**示例 1：**

```
输入：
Employee 表：
+----+--------+
| id | salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
输出：
+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
```

**示例 2：**

```
输入：
Employee 表：
+----+--------+
| id | salary |
+----+--------+
| 1  | 100    |
+----+--------+
输出：
+---------------------+
| SecondHighestSalary |
+---------------------+
| null                |
+---------------------+
```

通过次数331,613

提交次数924,651

### 解法一

~~~sql
select IFNULL((select distinct salary from Employee order by salary desc limit 1,1 ),NULL) SecondHighestSalary ;
~~~

### 解法二

~~~sql
SELECT
    (SELECT DISTINCT
            Salary
        FROM
            Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1) AS SecondHighestSalary
~~~



## [184. 部门工资最高的员工](https://leetcode.cn/problems/department-highest-salary/)

难度中等537收藏分享切换为英文接收动态反馈

SQL架构

表： `Employee`

```
+--------------+---------+
| 列名          | 类型    |
+--------------+---------+
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |
+--------------+---------+
id是此表的主键列。
departmentId是Department表中ID的外键。
此表的每一行都表示员工的ID、姓名和工资。它还包含他们所在部门的ID。
```

 

表： `Department`

```
+-------------+---------+
| 列名         | 类型    |
+-------------+---------+
| id          | int     |
| name        | varchar |
+-------------+---------+
id是此表的主键列。
此表的每一行都表示一个部门的ID及其名称。
```

 

编写SQL查询以查找每个部门中薪资最高的员工。
按 **任意顺序** 返回结果表。
查询结果格式如下例所示。

 

**示例 1:**

```
输入：
Employee 表:
+----+-------+--------+--------------+
| id | name  | salary | departmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Jim   | 90000  | 1            |
| 3  | Henry | 80000  | 2            |
| 4  | Sam   | 60000  | 2            |
| 5  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
Department 表:
+----+-------+
| id | name  |
+----+-------+
| 1  | IT    |
| 2  | Sales |
+----+-------+
输出：
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
| IT         | Max      | 90000  |
+------------+----------+--------+
解释：Max 和 Jim 在 IT 部门的工资都是最高的，Henry 在销售部的工资最高。
```

通过次数138,060

提交次数276,977

### 解法一

~~~sql
select d1.name Department, e1.name Employee, e1.salary
from Department d1,Employee e1,(select max(salary) sl,departmentId from Employee group by departmentId)e2
where e1.departmentId=d1.id and e1.departmentId=e2.departmentId and e1.salary=e2.sl
~~~



### 解法二

~~~sql
SELECT
    Department.name AS 'Department',
    Employee.name AS 'Employee',
    Salary
FROM
    Employee
        JOIN
    Department ON Employee.DepartmentId = Department.Id
WHERE
    (Employee.DepartmentId , Salary) IN
    (   SELECT
            DepartmentId, MAX(Salary)
        FROM
            Employee
        GROUP BY DepartmentId
	)


~~~



## [196. 删除重复的电子邮箱](https://leetcode.cn/problems/delete-duplicate-emails/)

难度简单557收藏分享切换为英文接收动态反馈

SQL架构

表: `Person`

```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| email       | varchar |
+-------------+---------+
id是该表的主键列。
该表的每一行包含一封电子邮件。电子邮件将不包含大写字母。
```

 

编写一个 SQL **删除语句**来 **删除** 所有重复的电子邮件，只保留一个id最小的唯一电子邮件。

以 **任意顺序** 返回结果表。 （**注意**： 仅需要写删除语句，将自动对剩余结果进行查询）

查询结果格式如下所示。

 

 

**示例 1:**

```
输入: 
Person 表:
+----+------------------+
| id | email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
| 3  | john@example.com |
+----+------------------+
输出: 
+----+------------------+
| id | email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
+----+------------------+
解释: john@example.com重复两次。我们保留最小的Id = 1。
```

通过次数155,727

提交次数229,810

### 直接求解

注意表如果有别名，删除的时候带上别名

~~~sql
DELETE p1 from Person p1
,(select min(id) min_id,email from Person 
group by email
)p2
where p1.email=p2.email and p1.id>p2.min_id
~~~



### 解法二

~~~sql
DELETE p1 FROM Person p1,
    Person p2
WHERE
    p1.Email = p2.Email AND p1.Id > p2.Id
~~~





## [177. 第N高的薪水](https://leetcode.cn/problems/nth-highest-salary/)

难度中等610收藏分享切换为英文接收动态反馈

SQL架构

表: `Employee`

```
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
Id是该表的主键列。
该表的每一行都包含有关员工工资的信息。
```

 

编写一个SQL查询来报告 `Employee` 表中第 `n` 高的工资。如果没有第 `n` 个最高工资，查询应该报告为 `null` 。

查询结果格式如下所示。

 

**示例 1:**

```
输入: 
Employee table:
+----+--------+
| id | salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
n = 2
输出: 
+------------------------+
| getNthHighestSalary(2) |
+------------------------+
| 200                    |
+------------------------+
```

**示例 2:**

```
输入: 
Employee 表:
+----+--------+
| id | salary |
+----+--------+
| 1  | 100    |
+----+--------+
n = 2
输出: 
+------------------------+
| getNthHighestSalary(2) |
+------------------------+
| null                   |
+------------------------+
```

通过次数170,255

提交次数364,988

### 解答

~~~sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      # Write your MySQL query statement below.
      select salary from
        (select  e1.salary, count(1) cnt
            from(select distinct salary from Employee) e1,(select distinct salary from Employee) e2
        where e1.salary<=e2.salary
        group by e1.salary
        having cnt=N
        )e3
       
      
  );
END
~~~



### 解法二

~~~sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      # Write your MySQL query statement below.
      SELECT 
          DISTINCT e.salary
      FROM 
          employee e
      WHERE 
          (SELECT count(DISTINCT salary) FROM employee WHERE salary>e.salary) = N-1
  );
END
~~~

